"""Event-driven backtester.

The backtester deliberately reuses the *live* objects -- :class:`RiskManager`,
:class:`PositionSizer`, :class:`PositionBook`, :class:`MetricTracker` and the same
:class:`Strategy` call -- rather than reimplementing the rules. A backtest that
disagrees with live behaviour is worse than no backtest, and the cheapest way to
guarantee agreement is to share the code.

Execution model, chosen to avoid the usual ways a backtest lies to you:

* **Decisions see closed bars only**, and orders are filled at the **next bar's
  open** (``execution_delay_bars=1``). Filling at the close that produced the
  signal is the classic look-ahead bias: you cannot trade a price you need in
  order to decide to trade.
* **Stops are intrabar.** A stop is detected against the bar's high/low and
  filled at the stop price, not the close. If a bar gaps through the stop the
  fill is the stop *plus slippage* -- which still understates a real gap, so
  treat stop-heavy strategies as optimistic.
* **Costs are charged on both legs**: ``taker_fee`` per side plus
  ``slippage_bps`` of adverse price. At 0.1% per side a round trip costs ~0.2%,
  which for a 30-minute mean-reversion system is the entire edge unless the
  average target move clears it. The report prints fees against gross P&L for
  exactly this reason.
* **Rule 1 still runs.** The venue's ticker is synthesised from the bars: the
  close is the mid, ``assumed_spread_bps`` sets the quote, and the 24h turnover
  ranking uses the rolling sum of quote volume. Since one assumed spread is
  applied to every pair, the spread *filter* cannot discriminate in a backtest --
  it is exercised, not validated. Depth is not available historically and the
  selector is built with no depth provider.
"""

from __future__ import annotations
from .strategies.scoring import execution_terms

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .candles import Candle, bar_index, load_candles
from .config import Config
from .journal import Journal
from .metrics import MetricTracker, PerformanceMetrics, compute_metrics, daily_equity_marks
from .models import Fill, Position, Ticker, fmt
from .risk import (
    ApprovedAction,
    PortfolioView,
    PositionBook,
    PositionSizer,
    RiskManager,
    portfolio_nav,
)
from .strategies.base import (
    ENTER_LONG,
    ENTER_SHORT,
    EXIT_LONG,
    EXIT_SHORT,
    MarketContext,
    Signal,
    Strategy,
)
from .universe import NullDepthProvider, UniverseSelector

log = logging.getLogger(__name__)

SHORT_FEE_RATE = 0.001  # the v6 short endpoints charge 0.1% regardless of maker/taker


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------


def slice_candles(
    candles: dict[str, list[Candle]],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> dict[str, list[Candle]]:
    """Keep bars in ``[start_ts, end_ts]`` inclusive."""
    out: dict[str, list[Candle]] = {}
    for pair, series in candles.items():
        out[pair] = [
            c
            for c in series
            if (start_ts is None or c.ts_ms >= start_ts) and (end_ts is None or c.ts_ms <= end_ts)
        ]
    return out


def split_timeline(candles: dict[str, list[Candle]], frac: float) -> tuple[int, int]:
    """Return ``(split_ts, last_ts)`` for an out-of-sample holdout."""
    timeline = sorted({c.ts_ms for series in candles.values() for c in series})
    if not timeline:
        raise ValueError("no candles to split")
    idx = max(1, min(len(timeline) - 1, int(len(timeline) * frac)))
    return timeline[idx], timeline[-1]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalise_reason(reason: str) -> str:
    """Collapse numbers so rejection reasons can be counted together."""
    return _NUMBER.sub("#", reason).strip()


@dataclass
class BacktestResult:
    label: str = "backtest"
    metrics: Optional[PerformanceMetrics] = None
    trades: list[Fill] = field(default_factory=list)
    equity: list[tuple[int, float]] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)
    per_pair: dict[str, dict[str, float]] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    bars: int = 0
    universe_events: int = 0
    final_cash: float = 0.0
    #: First timestamp that belongs to this window's *measurement* period.
    #: An out-of-sample slice is built with a warm-up prefix so the indicators are
    #: primed at the split, and those prefix bars must not enter the metrics: their
    #: flat marks diluted the OOS returns and shortened the annualisation period.
    metrics_from_ts: Optional[int] = None

    # -- derived --------------------------------------------------------
    @property
    def fees_paid(self) -> float:
        return sum(t.fee for t in self.trades)

    @property
    def gross_pnl(self) -> float:
        """Realised + unrealised price P&L, before fees."""
        return sum(t.quantity * t.price * (1 if t.side == "SELL" else -1) for t in self.trades)

    def report(self) -> str:
        lines: list[str] = []
        if self.metrics:
            lines.append(f"=== {self.label} ===")
            lines.extend(self.metrics.summary_lines())
        lines.append("")
        lines.append(f"bars processed    {self.bars}")
        lines.append(f"fills             {len(self.trades)}")
        lines.append(f"fees paid         {self.fees_paid:,.2f}")
        entries = sum(1 for t in self.trades if t.side == "BUY" or t.side == "SHORT_OPEN")
        lines.append(f"entries / exits   {entries} / {len(self.trades) - entries}")

        if self.per_pair:
            lines.append("")
            lines.append("per pair:")
            lines.append(f"  {'pair':<10} {'fills':>6} {'notional':>14} {'fees':>10}")
            for pair, stats in sorted(self.per_pair.items(), key=lambda kv: -kv[1]["notional"]):
                lines.append(
                    f"  {pair:<10} {int(stats['fills']):>6} {stats['notional']:>14,.2f} {stats['fees']:>10,.2f}"
                )

        if self.rejections:
            lines.append("")
            lines.append("top rejection reasons:")
            for reason, count in self.rejections.most_common(12):
                lines.append(f"  {count:>5}  {reason}")

        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            # Recorded so a reader can tell which part of `equity` the metrics
            # were actually computed from: an out-of-sample slice carries a
            # warm-up prefix that must not be counted as out-of-sample days.
            "metrics_from_ts": self.metrics_from_ts,
            "trades": len(self.trades),
            "fees_paid": round(self.fees_paid, 4),
            "bars": self.bars,
            "rejections": dict(self.rejections.most_common(40)),
            "per_pair": self.per_pair,
            "warnings": self.warnings,
            "final_cash": round(self.final_cash, 4),
            "equity": [[ts, round(v, 4)] for ts, v in self.equity],
        }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Replays candles through the live decision path."""

    def __init__(
        self,
        cfg: Config,
        strategy: Strategy,
        candles: dict[str, list[Candle]],
        execution_delay_bars: int = 1,
        assumed_spread_bps: float = 5.0,
        label: str = "backtest",
        trade_from_ts: Optional[int] = None,
        journal: Optional[Journal] = None,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        configure = getattr(self.strategy, "configure_execution", None)
        if configure is not None:
            configure(cfg)
        self.label = label
        self.execution_delay_bars = max(0, int(execution_delay_bars))
        self.assumed_spread_bps = float(assumed_spread_bps)
        self.trade_from_ts = trade_from_ts
        self.journal = journal

        self.series: dict[str, list[Candle]] = {p: sorted(s, key=lambda c: c.ts_ms) for p, s in candles.items() if s}
        if not self.series:
            raise ValueError("no candles supplied")
        self.by_ts: dict[str, dict[int, Candle]] = {p: {c.ts_ms: c for c in s} for p, s in self.series.items()}
        self.timeline: list[int] = sorted({c.ts_ms for s in self.series.values() for c in s})
        self.bar_ms = self.cfg.bar_seconds * 1000

        # Live-equivalent state.
        self.cash_usd = float(cfg.initial_capital)
        self.book = PositionBook(None)
        self.risk = RiskManager(cfg, PositionSizer(cfg))
        self.tracker = MetricTracker(cfg.initial_capital, periods_per_year=cfg.periods_per_year, risk_free_rate=cfg.risk_free_rate)
        self.selector = UniverseSelector(cfg, NullDepthProvider())

        self.history: dict[str, list[Candle]] = {p: [] for p in self.series}
        self.pending: dict[int, list[ApprovedAction]] = {}
        self.trades: list[Fill] = []
        self.rejections: Counter = Counter()
        self.universe: list[str] = []
        self.universe_events = 0
        self.result = BacktestResult(label=label)
        self._entry_price: dict[str, float] = {}

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        warmup = self.strategy.required_bars
        if self.journal:
            self.journal.startup(self.cfg.redacted(), self.strategy.describe(), extra={"mode": "backtest"})

        for ts in self.timeline:
            bars = {p: self.by_ts[p][ts] for p in self.series if ts in self.by_ts[p]}
            if not bars:
                continue

            self._append_history(bars)
            self._execute_pending(ts, bars)
            self._update_marks(bars)
            self._protective_exits(ts, bars)

            nav = portfolio_nav(self.cash_usd, self.book.positions)
            self.tracker.record(ts + self.bar_ms, nav)
            self.risk.observe(nav, ts + self.bar_ms)

            if self.risk.halted:
                self.result.warnings.append(f"kill switch tripped: {self.risk.halt_reason}")
                self._flatten_now(ts, bars)
                break

            if self.trade_from_ts is not None and ts < self.trade_from_ts:
                self.result.bars += 1
                continue

            self._decision(ts, bars, nav)
            self.result.bars += 1

        final_nav = portfolio_nav(self.cash_usd, self.book.positions)
        self.tracker.record(self.timeline[-1] + self.bar_ms, final_nav) if self.timeline else None
        equity = list(self.tracker.snapshots)
        # Measure only from `trade_from_ts`. An out-of-sample slice carries a
        # warm-up prefix so the indicators are primed at the split; those bars are
        # not part of the out-of-sample period, and counting their flat marks
        # diluted the OOS returns and mis-stated the annualisation period.
        measured = [s for s in equity if self.trade_from_ts is None or s[0] >= self.trade_from_ts] or equity
        self.result.metrics_from_ts = self.trade_from_ts
        daily = daily_equity_marks(measured)
        curve = daily if len(daily) >= 2 else [self.cfg.initial_capital] + [v for _, v in measured]
        self.result.metrics = compute_metrics(
            curve, periods_per_year=self.cfg.periods_per_year, risk_free_rate=self.cfg.risk_free_rate
        )
        self.result.equity = equity
        self.result.trades = self.trades
        self.result.rejections = self.rejections
        self.result.per_pair = self._per_pair_stats()
        self.result.diagnostics = dict(self.strategy.diagnostics)
        self.result.universe_events = self.universe_events
        self.result.final_cash = self.cash_usd
        self._validate(warmup, nav=final_nav)
        if self.journal:
            self.journal.close()
        return self.result

    # ------------------------------------------------------------------
    # Per-bar stages
    # ------------------------------------------------------------------
    def _append_history(self, bars: dict[str, Candle]) -> None:
        for pair, candle in bars.items():
            self.history.setdefault(pair, []).append(candle)
            if len(self.history[pair]) > self.cfg.history_window:
                del self.history[pair][0]

    def _execute_pending(self, ts: int, bars: dict[str, Candle]) -> None:
        actions = self.pending.pop(ts, [])
        for action in actions:
            candle = bars.get(action.pair)
            if candle is None:
                # The pair has no bar at this timestamp (a gap). Re-queue once.
                later = self._next_ts(ts)
                if later is not None:
                    self.pending.setdefault(later, []).append(action)
                else:
                    self.result.warnings.append(f"dropped {action.action} {action.pair}: no bar to fill against")
                continue
            self._fill(action, reference_price=candle.open, ts=ts, source="open")

    def _update_marks(self, bars: dict[str, Candle]) -> None:
        for pair, position in self.book.positions.items():
            candle = bars.get(pair)
            if candle is not None:
                position.update_mark(candle.close)

    def _protective_exits(self, ts: int, bars: dict[str, Candle]) -> None:
        """Rule 5/6 detection via the live risk layer, filled at the bar's extremes."""
        if not self.book.positions:
            return
        worst: dict[str, Ticker] = {}
        for pair, position in self.book.positions.items():
            candle = bars.get(pair)
            if candle is None:
                continue
            # The most adverse price the bar actually traded for this direction.
            extreme = candle.high if position.is_short else candle.low
            worst[pair] = Ticker(
                pair=pair,
                last=extreme,
                max_bid=extreme,
                min_ask=extreme,
                change_24h=0.0,
                coin_volume=0.0,
                unit_volume=0.0,
                server_time_ms=ts,
            )
        if not worst:
            return

        signals = self.risk.protective_exits(self.book.held(), worst, ts + self.bar_ms)
        for signal in signals:
            candle = bars.get(signal.pair)
            position = self.book.get(signal.pair)
            if candle is None or position is None:
                continue
            trigger = signal.meta.get("trigger")
            if trigger == "stop":
                # A stop order fills at its level, not at the extreme that hit it.
                # Slippage is still applied, but a genuine gap through the stop
                # will be understated -- noted in the module docstring.
                reference = float(position.stop_price or candle.close)
            else:
                reference = candle.close
            self._fill(
                signal_to_action(signal, position),
                reference_price=reference,
                ts=ts,
                source=str(trigger),
                ts_ms=ts + self.bar_ms,
            )

    def _decision(self, ts: int, bars: dict[str, Candle], nav: float) -> None:
        tickers = self._synthetic_tickers(bars)
        self._refresh_universe(ts, tickers)

        # Bound the history handed to the strategy; see Strategy.max_context_bars.
        context_bars = max(1, self.strategy.max_context_bars)
        ctx = MarketContext(
            now_ms=ts + self.bar_ms,
            bar_seconds=self.cfg.bar_seconds,
            candles={p: self.history.get(p, [])[-context_bars:] for p in self.history},
            tickers=tickers,
            nav=nav,
            cash_usd=self.cash_usd,
            positions=self.book.held(),
            universe=list(self.universe),
            blocked=self.risk.blocked_pairs(bar_index(ts + self.bar_ms, self.cfg.bar_seconds)),
            daily_halt=self.risk.daily_halt,
            bar_index=bar_index(ts + self.bar_ms, self.cfg.bar_seconds),
            is_backtest=True,
        )
        signals = self.strategy.generate(ctx)
        view = PortfolioView(nav=nav, cash_usd=self.cash_usd, positions=self.book.held())
        # Orders already queued for a later bar are not positions yet, but they
        # must reserve capital and slots or consecutive bars would each size a
        # full position for the same pair.
        in_flight = [a for actions in self.pending.values() for a in actions]
        decision = self.risk.evaluate(
            signals,
            view=view,
            tickers=tickers,
            now_ms=ts + self.bar_ms,
            bar_idx=bar_index(ts + self.bar_ms, self.cfg.bar_seconds),
            committed_pairs={a.pair for a in in_flight},
            committed_notional=sum(a.notional for a in in_flight if a.is_entry),
        )
        for pair, reason in decision.rejected:
            self.rejections[normalise_reason(reason)] += 1
        if self.journal:
            self.journal.signals(ts + self.bar_ms, signals, diagnostics=self.strategy.diagnostics)
            self.journal.decision(ts + self.bar_ms, decision.to_dict())

        if self.execution_delay_bars == 0:
            # Optimistic mode: fill against the close that produced the signal.
            # Only useful for comparing against the delay-1 results.
            for action in decision.approved:
                candle = bars.get(action.pair)
                if candle is not None:
                    self._fill(action, reference_price=candle.close, ts=ts, source="close")
            return

        # An action decided on bar t is filled at the open of bar t + delay.
        target = ts + self.execution_delay_bars * self.bar_ms
        for action in decision.approved:
            self.pending.setdefault(target, []).append(action)

    def _refresh_universe(self, ts: int, tickers: dict[str, Ticker]) -> None:
        need = self.strategy.required_bars
        candidates = {p: t for p, t in tickers.items() if len(self.history.get(p, [])) >= need}
        selection = self.selector.select(
            candidates,
            required_notional=self.cfg.depth_target_notional,
            can_trade=candidates.keys(),
            previous=self.universe or self.book.positions.keys(),
        )
        self.universe = selection.selected
        self.universe_events += 1
        if self.journal:
            self.journal.universe(ts + self.bar_ms, selection.to_dict())

    def _synthetic_tickers(self, bars: dict[str, Candle]) -> dict[str, Ticker]:
        """Reconstruct the venue ticker from bars, so Rule 1 can run offline.

        Quote volume over the trailing 24h is the turnover proxy; the spread is an
        assumption because the CSV has no book. ``unit_volume`` is therefore a
        *lower bound* used only for ranking.
        """
        half_spread = self.assumed_spread_bps / 2.0 / 10_000.0
        window = max(1, int(self.cfg.bars_per_day()))
        out: dict[str, Ticker] = {}
        for pair, candle in bars.items():
            series = self.history.get(pair, [])
            trailing = series[-window:] if series else [candle]
            turnover = sum(c.quote_volume for c in trailing)
            if turnover <= 0:
                turnover = sum(c.volume * c.close for c in trailing)
            out[pair] = Ticker(
                pair=pair,
                last=candle.close,
                max_bid=candle.close * (1.0 - half_spread),
                min_ask=candle.close * (1.0 + half_spread),
                change_24h=0.0,
                coin_volume=sum(c.volume for c in trailing),
                unit_volume=turnover,
                server_time_ms=candle.ts_ms,
            )
        return out

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def _fill(
        self,
        action: ApprovedAction,
        reference_price: float,
        ts: int,
        source: str,
        ts_ms: Optional[int] = None,
    ) -> None:
        """Apply one action at ``reference_price`` with slippage and fees.

        ``ts`` is the timestamp of the bar whose price was used, and the recorded
        fill time is derived from it. Stamping every fill with ``ts + bar_ms``
        labelled each one with the *next* bar's close: an order decided on bar
        *t* and filled at the open of *t + 1* was journalled 30 minutes later
        than it happened, so ``trades_*.csv`` -- the artefact
        ``scripts/analyze_backtest.py`` reads -- did not line up with the bars.
        A caller that genuinely acts at the close (a stop or an emergency
        flatten detected on the bar's close) may pass ``ts_ms`` explicitly.
        """
        if reference_price <= 0:
            return
        slip = self.cfg.slippage_bps / 10_000.0
        if ts_ms is None:
            ts_ms = ts

        if action.action == ENTER_LONG:
            if "scoring_version" in action.meta:
                rejection, terms = execution_terms(
                    self.cfg, action.meta, reference=reference_price,
                    spread_bps=self.assumed_spread_bps,
                    stop_price=action.stop_price, now_ms=ts_ms,
                )
                if rejection:
                    self.rejections[rejection] += 1
                    return
                # A next-open gap can change stop risk. Shrink, never enlarge,
                # the approved quantity using the current NAV and fixed stop.
                fill_price = reference_price * (1.0 + slip)
                stop_loss = fill_price - action.stop_price
                nav_now = portfolio_nav(self.cash_usd, self.book.held())
                loss_per_unit = stop_loss + fill_price * terms["cost_pct"]
                quantity_cap = nav_now * terms["risk_per_trade_pct"] / loss_per_unit
                action.quantity = min(action.quantity, quantity_cap)
                action.notional = action.quantity * fill_price
                action.risk_amount = action.quantity * loss_per_unit
                action.meta.update(terms)
                if action.notional < self.cfg.min_order_notional:
                    self.rejections["scored fill below minimum notional"] += 1
                    return
                  
            price = reference_price * (1.0 + slip)
            quantity = self._clamp_quantity(action.quantity, price, self.cash_usd)
            if quantity <= 0:
                self.rejections["insufficient cash at fill time"] += 1
                return
            notional = quantity * price
            fee = notional * self.cfg.taker_fee
            if notional + fee > self.cash_usd:
                self.rejections["insufficient cash for fee"] += 1
                return
            self.cash_usd -= notional + fee
            self.book.apply_spot_buy(action.pair, quantity, price, ts_ms, action.stop_price)
            self._entry_price[action.pair] = price
            self._record_trade(ts_ms, action, "BUY", quantity, price, fee, "TAKER")

        elif action.action == EXIT_LONG:
            position = self.book.get(action.pair)
            if position is None or position.quantity <= 0:
                return
            price = reference_price * (1.0 - slip)
            quantity = position.quantity
            proceeds = quantity * price
            fee = proceeds * self.cfg.taker_fee
            self.cash_usd += proceeds - fee
            self.book.apply_spot_sell(action.pair, quantity, price)
            self.risk.record_exit(action.pair, bar_index(ts_ms, self.cfg.bar_seconds))
            self._record_trade(ts_ms, action, "SELL", quantity, price, fee, "TAKER")

        elif action.action == ENTER_SHORT:
            price = reference_price * (1.0 - slip)  # a short sells into the bid
            collateral = min(action.collateral, max(0.0, self.cash_usd - 1.0))
            if collateral < 1.0:
                self.rejections["insufficient cash for short collateral"] += 1
                return
            fee = collateral * SHORT_FEE_RATE
            if collateral + fee > self.cash_usd:
                self.rejections["insufficient cash for short fee"] += 1
                return
            quantity = float(fmt(collateral / price, 6))
            if quantity <= 0:
                return
            self.cash_usd -= fee  # the collateral stays inside cash_total, as live
            self.book.apply_short_open(action.pair, quantity, price, collateral, ts_ms, action.stop_price)
            self._record_trade(ts_ms, action, "SHORT_OPEN", quantity, price, fee, "TAKER")

        elif action.action == EXIT_SHORT:
            position = self.book.get(action.pair)
            if position is None or position.quantity <= 0:
                return
            price = reference_price * (1.0 + slip)  # buying the short back pays the ask
            quantity = position.quantity
            realized = quantity * (position.avg_price - price)
            realized = max(realized, -position.collateral)  # cannot lose more than collateral
            fee = quantity * price * SHORT_FEE_RATE
            self.cash_usd += realized - fee
            self.book.apply_short_close(action.pair, quantity, price)
            self.risk.record_exit(action.pair, bar_index(ts_ms, self.cfg.bar_seconds))
            self._record_trade(ts_ms, action, "SHORT_CLOSE", quantity, price, fee, "TAKER")

    def _record_trade(
        self, ts_ms: int, action: ApprovedAction, side: str, quantity: float, price: float, fee: float, role: str
    ) -> None:
        fill = Fill(
            ts_ms=ts_ms,
            pair=action.pair,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            order_id=None,
            role=role,
            reason=action.reason,
        )
        self.trades.append(fill)
        if self.journal:
            self.journal.trade(
                ts_ms, action.pair, action.action, side, quantity, price, fee, "", role, action.reason
            )

    @staticmethod
    def _clamp_quantity(quantity: float, price: float, cash: float) -> float:
        """Never spend cash we do not have."""
        if price <= 0 or cash <= 0:
            return 0.0
        affordable = cash / price
        return max(0.0, min(quantity, affordable))

    def _flatten_now(self, ts: int, bars: dict[str, Candle]) -> None:
        for action in self.risk.flatten_all(self.book.held()):
            candle = bars.get(action.pair)
            if candle is None:
                continue
            # The halt was detected on this bar's close, so the flatten is booked
            # at the close of the bar it happened on, not the one before it.
            self._fill(
                action, reference_price=candle.close, ts=ts, source="kill_switch", ts_ms=ts + self.bar_ms
            )

    def _next_ts(self, ts: int) -> Optional[int]:
        try:
            idx = self.timeline.index(ts)
        except ValueError:
            return None
        return self.timeline[idx + 1] if idx + 1 < len(self.timeline) else None

    # ------------------------------------------------------------------
    def _per_pair_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for fill in self.trades:
            row = stats.setdefault(fill.pair, {"fills": 0.0, "notional": 0.0, "fees": 0.0})
            row["fills"] += 1
            row["notional"] += fill.notional
            row["fees"] += fill.fee
        return {p: {k: round(v, 4) for k, v in row.items()} for p, row in stats.items()}

    def _validate(self, warmup: int, nav: float) -> None:
        """Cheap self-checks that catch the usual silent backtest failures."""
        if not self.trades:
            self.result.warnings.append(
                "no trades were executed: loosen the filters or check that the data spans "
                f"enough bars (warm-up needs {warmup} bars per pair)"
            )
        if self.result.bars <= warmup:
            self.result.warnings.append(
                f"only {self.result.bars} bars processed, barely above the {warmup}-bar warm-up"
            )
        if self.result.fees_paid > abs(self.result.gross_pnl) and self.trades:
            self.result.warnings.append(
                f"fees ({self.result.fees_paid:,.0f}) exceed gross P&L ({self.result.gross_pnl:,.0f}): "
                "the strategy is trading more than the edge supports"
            )
        largest = max((t.notional for t in self.trades), default=0.0)
        if largest > self.cfg.max_pair_weight * self.cfg.initial_capital * 1.5:
            self.result.warnings.append(
                f"largest fill {largest:,.0f} is far above the {self.cfg.max_pair_weight:.0%} per-pair cap; "
                "check the sizing path"
            )
        if not math.isfinite(nav):
            self.result.warnings.append("final NAV is not finite")
        if self.strategy.diagnostics:
            skipped = Counter(
                normalise_reason(str(v))
                for row in self.strategy.diagnostics.values()
                for k, v in row.items()
                if k == "entry_skipped"
            )
            self.result.diagnostics["_entry_skip_reasons"] = dict(skipped.most_common(10))


def signal_to_action(signal: Signal, position: Position) -> ApprovedAction:
    """Adapt a protective-exit signal into an executable action."""
    return ApprovedAction(
        pair=signal.pair,
        action=signal.action,
        quantity=position.quantity,
        collateral=position.collateral,
        notional=abs(position.notional),
        reason=signal.reason,
        meta=dict(signal.meta),
    )


def load_universe_candles(
    data_dir: str,
    pairs: Iterable[str],
    interval: str = "30m",
    bar_seconds: int = 1800,
) -> dict[str, list[Candle]]:
    """Load ``<PAIR>_<interval>.csv`` files from ``data_dir``.

    ``BTC/USD`` maps to ``BTC-USD_30m.csv``, matching
    ``scripts/fetch_history.py``.
    """
    from pathlib import Path

    base = Path(data_dir)
    out: dict[str, list[Candle]] = {}
    for pair in pairs:
        slug = pair.replace("/", "-")
        candidates = [base / f"{slug}_{interval}.csv", base / f"sample_{slug}_{interval}.csv"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            log.warning("no candle file for %s (looked for %s)", pair, [str(c) for c in candidates])
            continue
        series = load_candles(path, bar_seconds=bar_seconds)
        if series:
            out[pair] = series
            log.info("loaded %d bars for %s from %s", len(series), pair, path)
    return out
