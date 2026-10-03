"""The autonomous decision loop.

Design constraints this file has to satisfy at once:

1. **Rules 2-3 are defined on closed 30-minute bars**, so the strategy must run
   exactly once per bar -- not once per loop.
2. **Rule 5's stop is a price level**, so protective exits must be checked on
   every loop, not only at the bar boundary.
3. **Exits must be more reliable than entries.** Flattening has to work even when
   the risk layer is halting new business.
4. **A 14-day unattended run will restart.** Position state is persisted, and on
   every cycle the local book is reconciled against the exchange's balances --
   the exchange is the source of truth for quantity.
5. **An order that fails in transport has an unknown outcome.** It is never
   blind-retried; it is reconciled against the order history.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .basis import BasisMonitor
from .candles import Candle, CandleBuilder, bar_index, load_candles, trading_day_id
from .client import build_client, is_success
from .config import Config
from .journal import Journal
from .metrics import MetricTracker
from .models import OrderResult, Position, Ticker, TradePair, WalletBalance, fmt
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
    load_strategy,
)
from .universe import UniverseSelection, UniverseSelector, build_depth_provider

log = logging.getLogger(__name__)

#: If the seeded history's last close is further than this from the live venue's
#: price, the two feeds disagree and the seeded bars would poison every z-score.
#:
#: The organisers confirmed the venue tracks Binance, so a gap this large is not
#: "a different but related market" -- it is a wrong symbol, a wrong quote
#: currency, or a stale feed. Kept deliberately tight for that reason; the
#: previous 2% allowance would have let a genuine USDT/USD mismatch through.
SEED_TOLERANCE_PCT = 0.005

#: Outcomes of `TradingEngine._reconcile_unknown_order`.
#:
#: ``ORDER_UNKNOWN`` must not be released: a row can be missing because the venue
#: has not written it yet, and assuming "not found" means "not filled" is how a
#: position arrives later with nothing protecting it.
ORDER_FILLED = "filled"
ORDER_NOT_FILLED = "not_filled"
ORDER_UNKNOWN = "unknown"

#: Venue statuses that settle an order as gone without a fill.
TERMINAL_NOT_FILLED = frozenset({"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"})


@dataclass
class EngineStats:
    cycles: int = 0
    bars_processed: int = 0
    entries: int = 0
    exits: int = 0
    orders_sent: int = 0
    order_errors: int = 0
    unknown_orders: int = 0
    reconciliations: int = 0
    consecutive_failures: int = 0
    #: Maker entries posted and cancelled. The ratio is the live fill rate, which
    #: is the number the backtest's 95-99% assumption most needs checking against.
    orders_posted: int = 0
    orders_cancelled: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "cycles": self.cycles,
            "bars_processed": self.bars_processed,
            "entries": self.entries,
            "exits": self.exits,
            "orders_sent": self.orders_sent,
            "order_errors": self.order_errors,
            "unknown_orders": self.unknown_orders,
            "reconciliations": self.reconciliations,
            "orders_posted": self.orders_posted,
            "orders_cancelled": self.orders_cancelled,
        }


class TradingEngine:
    """Live loop. ``step()`` is one cycle; ``run()`` repeats it."""

    def __init__(
        self,
        cfg: Config,
        client: Any = None,
        journal: Optional[Journal] = None,
        seed_paths: Optional[dict[str, str | Path]] = None,
    ) -> None:
        self.cfg = cfg
        self.client = client or build_client(cfg)
        self.journal = journal if journal is not None else Journal(cfg.journal_dir)
        self.strategy = load_strategy(cfg.strategy, cfg.strategy_params)
        self.stats = EngineStats()

        self.exchange_pairs: dict[str, TradePair] = {}
        self.tickers: dict[str, Ticker] = {}
        self.balances: dict[str, WalletBalance] = {}
        self.builder = CandleBuilder(bar_seconds=cfg.bar_seconds, max_bars=cfg.history_window)
        self.book = PositionBook(Path(cfg.journal_dir) / cfg.state_dir_name / "positions.json", cfg=cfg)
        self.sizer = PositionSizer(cfg)
        self.risk = RiskManager(cfg, self.sizer)
        self.tracker = MetricTracker(cfg.initial_capital, periods_per_year=cfg.periods_per_year, risk_free_rate=cfg.risk_free_rate)
        self.selector: Optional[UniverseSelector] = None
        self.universe: list[str] = []
        self.selection: Optional[UniverseSelection] = None

        self._last_decision_bar: Optional[int] = None
        self._seed_paths: dict[str, str | Path] = dict(seed_paths or {})
        self._pending_by_pair: dict[str, int] = {}
        #: Best-known notional per pair with a resting order, so the risk layer
        #: reserves what is actually committed rather than a flat full slot.
        self._pending_notional_by_pair: dict[str, float] = {}
        #: Maker entries we posted that have not filled yet, as
        #: ``pair -> (bar placed, limit price)``. Only used to cancel stale ones.
        self._resting: dict[str, tuple[int, float]] = {}
        self._shutting_down = False
        #: The halt journal entry is written once per run, not once per cycle:
        #: a halted engine keeps looping until its book is flat (see `step`).
        self._halt_announced = False
        #: Consecutive cycles that ended with positions still open after a halt.
        self._halt_flatten_failures = 0
        #: Orders whose outcome is still unknown, keyed by pair. Persisted: a
        #: transport failure is exactly the case where the process may be
        #: restarted before the venue can be asked again, and an order that is
        #: forgotten is a position that arrives later without its stop.
        self._unknown_intents: dict[str, dict[str, Any]] = {}
        #: Gate for ``_persist()``. Stays False until ``bootstrap()`` has
        #: completed, so a startup failure can never overwrite stored state.
        self._ready = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        """Sync the clock, learn the venue's rules, restore state, warm up."""
        # Restore persisted state FIRST, before anything that can fail on the
        # network. `run_live.py` always calls `shutdown()` from a `finally:`,
        # and `shutdown()` persists; if `sync_time()` or `exchange_info()` threw
        # before the book had been loaded, that persist would write a
        # default-empty book and default risk state over the real ones -- losing
        # every stop level, cost basis and cooldown, resetting the drawdown
        # high-water mark, and silently clearing the kill switch.
        self._migrate_legacy_state()
        self._load_unknown_intents()
        self.book.load()
        self._load_risk_state()

        offset = self.client.sync_time()
        info = self.client.exchange_info()
        self.exchange_pairs = {p: tp for p, tp in info.pairs.items() if tp.can_trade}
        if not info.is_running:
            raise RuntimeError("exchange reports IsRunning=false; refusing to trade")
        # Give the position book the venue's lot sizes, so "this row holds
        # nothing" is decided on the same threshold reconciliation uses.
        self.cfg._lot_resolver = lambda pair: 10.0 ** (-self.exchange_pairs[pair].amount_precision)

        configured = self.cfg.resolved_pairs()
        self.universe = [p for p in configured if p in self.exchange_pairs] if configured else []
        if configured and not self.universe:
            log.warning("none of the configured pairs are tradable; falling back to auto-discovery")

        self.journal.startup(
            self.cfg.redacted(),
            self.strategy.describe(),
            extra={
                "clock_offset_ms": offset,
                "tradable_pairs": sorted(self.exchange_pairs),
                "configured_universe": list(self.universe),
                "restored_positions": sorted(self.book.positions),
            },
        )
        log.info(
            "bootstrapped: %d tradable pairs, %d restored position(s), clock offset %+dms",
            len(self.exchange_pairs),
            len(self.book.positions),
            offset,
        )
        self._ready = True

    def seed_history(self) -> None:
        """Warm the indicators from CSV, but only if the feed agrees on price.

        Seeding matters: Rules 2-3 need 48 bars, i.e. 24 hours of uptime, before
        the very first signal. The prep window exists for this.

        It is also dangerous. The CSV is real market history while the venue is a
        mock book, so if their price levels disagree, gluing them together
        fabricates a deviation that never happened and the first trades chase a
        phantom. So: seed only when the last historical close is within
        ``SEED_TOLERANCE_PCT`` of the live mid, and say so in the journal either
        way.
        """
        tickers = self.client.ticker()
        for pair, path in self._seed_paths.items():
            if pair not in self.exchange_pairs:
                continue
            ticker = tickers.get(pair)
            if ticker is None or ticker.mid <= 0:
                self.journal.event("seed_skipped", pair=pair, reason="no live quote")
                continue
            try:
                candles = load_candles(path, bar_seconds=self.cfg.bar_seconds)
            except Exception as exc:
                self.journal.event("seed_skipped", pair=pair, reason=f"unreadable: {exc}")
                continue
            if not candles:
                self.journal.event("seed_skipped", pair=pair, reason="empty file")
                continue
            basis = ticker.mid / candles[-1].close - 1.0
            if abs(basis) > SEED_TOLERANCE_PCT:
                self.journal.event(
                    "seed_rejected",
                    pair=pair,
                    reason="history disagrees with venue price",
                    basis=round(basis, 6),
                    history_close=candles[-1].close,
                    venue_mid=ticker.mid,
                )
                log.warning(
                    "%s: refusing to seed, venue mid %.8f vs history close %.8f (basis %+.2f%%)",
                    pair,
                    ticker.mid,
                    candles[-1].close,
                    basis * 100,
                )
                continue
            self.builder.seed(pair, candles)
            self.journal.event(
                "seed_applied",
                pair=pair,
                bars=len(candles),
                basis=round(basis, 6),
                last_close=candles[-1].close,
                venue_mid=ticker.mid,
            )
            log.info("%s: seeded %d bars from %s", pair, len(candles), path)
        self.strategy.prepare(self._context(0))

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------
    def step(self) -> None:
        now_ms = int(time.time() * 1000)
        tickers = self.client.ticker()
        self.tickers = tickers
        balances = self.client.balance()

        # A partial or malformed balance snapshot must never be acted on. With no
        # USD row the portfolio cannot be priced: NAV collapses to the marks of
        # the positions alone, which reads as a catastrophic drawdown and trips
        # the *permanent* kill switch. The same payload would also make
        # reconciliation read every missing row as "the exchange holds nothing"
        # and delete the entire book. Skipping one cycle is cheap; acting on a
        # bad response liquidates the account's memory.
        if not self._balances_usable(balances):
            log.error("balance payload has no USD row (assets=%s); skipping this cycle", sorted(balances))
            self.journal.error(
                "cycle",
                "balance payload has no USD row; skipping rather than acting on an incomplete snapshot",
                ts_ms=now_ms,
                assets=sorted(balances),
            )
            return

        self.balances = balances
        shorts = self._safe_short_positions()

        self._reconcile_positions(balances, shorts)
        # Settle any order whose outcome was still unknown, before the book is
        # marked and before anything sizes against it.
        self._resolve_unknown_intents(now_ms, bar_index(now_ms, self.cfg.bar_seconds))
        self.book.mark(tickers)
        self._feed_candles(tickers, now_ms)

        cash_usd = self._cash_usd(balances)
        nav = portfolio_nav(cash_usd, self.book.positions)
        self.tracker.record(now_ms, nav)
        self.risk.observe(nav, now_ms)
        self.stats.cycles += 1

        view = PortfolioView(nav=nav, cash_usd=cash_usd, positions=self.book.held())
        depth_dump = None
        if self.selection is not None:
            depth_dump = self.selection.to_dict().get("depth")

        current_bar = bar_index(now_ms, self.cfg.bar_seconds)
        on_bar = current_bar != self._last_decision_bar

        self.journal.cycle(current_bar, now_ms, nav, cash_usd, depth=depth_dump)

        # --- kill switch: stop opening, then work the account flat ---------
        # "Halted" is not "finished". Flattening can fail, a cancel can fail, and
        # the exit status tells systemd this is a clean stop -- so a process that
        # exits while anything is still live leaves an order or a position with
        # nothing managing it and nothing that will restart it. Two ways that used
        # to happen: a flat book with a resting entry order exited without
        # cancelling it, and an un-expired resting order was never cancelled at
        # all because only the staleness sweep ran.
        #
        # So: cancel EVERY entry order, work the book flat, and require the account
        # to be genuinely clean -- no position, no tracked order, no venue order,
        # no unresolved intent -- before ending the run. An unanswered query counts
        # as "not clean", because assuming a venue holds nothing is how an order
        # gets orphaned. No entry can happen meanwhile: this branch returns before
        # any of the logic below.
        if self.risk.halted:
            if not self._halt_announced:
                self.journal.halt(now_ms, self.risk.halt_reason, nav=round(nav, 4))
                self._halt_announced = True

            self._cancel_all_entry_orders(now_ms)

            if self.book.held():
                self._flatten(now_ms, reason=self.risk.halt_reason)

            outstanding = self._halt_outstanding()
            if outstanding:
                self._halt_flatten_failures += 1
                # Log the first failure and then every tenth, so a venue outage
                # over a 14-day run is visible without filling the disk.
                if self._halt_flatten_failures == 1 or self._halt_flatten_failures % 10 == 0:
                    log.error(
                        "halted but not finished: %s. Keeping the loop alive rather than "
                        "exiting, because the exit status tells systemd not to restart "
                        "(attempt %d)",
                        outstanding,
                        self._halt_flatten_failures,
                    )
                    self.journal.event(
                        "halt_incomplete",
                        attempts=self._halt_flatten_failures,
                        outstanding=outstanding,
                        reason=self.risk.halt_reason,
                    )
                self._persist()
                return

            self._halt_flatten_failures = 0
            self._persist()
            self._shutting_down = True
            return

        # --- stale maker entries are released before new ones are posted -----
        # Cancelling first means a bid that has gone stale frees its capital and
        # its slot in the same cycle that would re-post it, instead of one bar later.
        self._expire_resting_orders(now_ms, current_bar)

        # --- protective exits run every loop (Rule 5) ---------------------
        protective = self.risk.protective_exits(self.book.held(), tickers, now_ms)
        if protective:
            self.journal.signals(now_ms, protective, diagnostics={"source": "risk.protective_exits"})
            self._refresh_pending()
            decision = self.risk.evaluate(
                protective,
                view=view,
                tickers=tickers,
                now_ms=now_ms,
                bar_idx=current_bar,
                committed_pairs=set(self._pending_by_pair),
                committed_notional=self._pending_notional(view.nav),
            )
            self._execute(decision.approved, now_ms, current_bar)
            # The exits above changed the book (and possibly the cash), so the
            # `view` built earlier in this cycle is stale. Rebuild it before the
            # strategy sees it, or a position closed by a stop this loop is still
            # reported as open and still consumes a Rule 8/9/10 slot.
            view = PortfolioView(
                nav=nav, cash_usd=cash_usd, positions=self.book.held()
            )

        # --- the strategy runs once per closed bar ------------------------
        if on_bar:
            self.stats.bars_processed += 1
            self._decision_cycle(now_ms, current_bar, view)
            # Stamped only after the cycle's work. Stamping before it meant an
            # exception inside the decision (swallowed by run()) consumed the bar
            # as if it had been evaluated, so that bar was never acted on.
            self._last_decision_bar = current_bar

        self.journal.equity(now_ms, nav, metrics=self.tracker.metrics().to_dict())
        self._persist()

    def _decision_cycle(self, now_ms: int, current_bar: int, view: PortfolioView) -> None:
        self._refresh_universe(now_ms)
        self._refresh_pending()
        ctx = self._context(now_ms, view)
        signals = self.strategy.generate(ctx)
        self.journal.signals(now_ms, signals, diagnostics=self.strategy.diagnostics)

        decision = self.risk.evaluate(
            signals,
            view=view,
            tickers=self.tickers,
            now_ms=now_ms,
            bar_idx=current_bar,
            committed_pairs=set(self._pending_by_pair),
            committed_notional=self._pending_notional(view.nav),
        )
        self.journal.decision(now_ms, decision.to_dict())
        self._execute(decision.approved, now_ms, current_bar)

    def _refresh_pending(self) -> None:
        """Learn which pairs have resting orders, so they can be reserved.

        A pending order is not a position yet, but its capital is committed. If
        the risk layer cannot see it, two consecutive bars will each size a full
        position for the same pair and the account ends up at twice the cap.
        """
        try:
            total, by_pair = self.client.pending_count()
            self._pending_by_pair = dict(by_pair) if total else {}
        except Exception as exc:
            log.debug("pending_count unavailable: %s", exc)
            self._pending_by_pair = {}
        # An order with an unknown outcome keeps its reservation even when the
        # venue does not list it. The outcome is unknown, not absent: releasing
        # the slot here would let the risk layer commit the same money twice on an
        # order that may already have filled.
        for pair in self._unknown_intents:
            self._pending_by_pair[pair] = max(int(self._pending_by_pair.get(pair, 0)), 1)
        self._pending_notional_by_pair = self._pending_notionals(self._pending_by_pair)
        for pair, intent in self._unknown_intents.items():
            self._pending_notional_by_pair[pair] = max(
                float(self._pending_notional_by_pair.get(pair, 0.0)),
                float(intent.get("notional") or 0.0),
            )

    def _pending_notionals(self, by_pair: dict[str, int]) -> dict[str, float]:
        """Per-pair committed notional for resting orders, best effort.

        ``pending_count`` reports how many orders rest per pair but not their
        size, so the exact figure comes from the order history. If that query
        fails we fall back to one slot per pair -- an over-estimate, which is the
        safe direction: it can only make the engine more conservative.
        """
        out: dict[str, float] = {}
        if not by_pair:
            return out
        try:
            rows = self.client.query_orders(pending_only=True, limit=100)
        except Exception as exc:
            log.debug("pending order detail unavailable (%s); reserving one slot per pair", exc)
            return out
        for row in rows or []:
            pair = str(row.get("Pair", "") or "")
            if pair not in by_pair:
                continue
            try:
                quantity = float(row.get("Quantity", 0.0) or 0.0)
                price = float(row.get("Price", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                ticker = self.tickers.get(pair)
                price = ticker.mid if ticker is not None else 0.0
            if quantity > 0 and price > 0:
                out[pair] = out.get(pair, 0.0) + quantity * price
        return out

    def _pending_notional(self, nav: float) -> float:
        """Total notional reserved by orders that are in flight but not filled."""
        if not self._pending_by_pair:
            return 0.0
        slot = self.sizer.slot_notional(nav)
        total = 0.0
        for pair in self._pending_by_pair:
            total += self._pending_notional_by_pair.get(pair, slot)
        return total

    # ------------------------------------------------------------------
    # Universe (Rule 1)
    # ------------------------------------------------------------------
    def _refresh_universe(self, now_ms: int) -> None:
        can_trade = set(self.exchange_pairs)
        provider = build_depth_provider(self.cfg, self.tickers, self.client)
        if self.selector is None or type(provider) is not type(self.selector.depth_provider):
            self.selector = UniverseSelector(self.cfg, provider)
        else:
            self.selector.depth_provider = provider

        configured = self.cfg.resolved_pairs()
        if configured:
            # An explicit list is a human decision: rank and filter only within it.
            candidates = {p: t for p, t in self.tickers.items() if p in configured}
        else:
            candidates = dict(self.tickers)

        # Cross-venue basis check (once per bar, one HTTP call for the universe).
        # The organisers confirmed the venue tracks Binance, so a wide basis is not
        # "a related but different market" -- it is the wrong symbol, the wrong
        # quote currency, or a stale feed, and a z-score computed against it is
        # noise. This fails open by design: a Binance outage must not stop the bot
        # from trading its own sampled bars, but the readings are journalled so a
        # drift shows up as a time series rather than a silent assumption.
        #
        # Skipped against the simulator, whose prices are synthetic by
        # construction: comparing them with Binance would block every pair.
        if not self.cfg.mock:
            basis_report = BasisMonitor(timeout=self.cfg.request_timeout_sec).check(candidates)
            self.journal.event("basis", ts_ms=now_ms, **basis_report.to_dict())
            blocked = basis_report.blocked()
            if blocked:
                log.warning("basis out of tolerance on %s; excluding them for this bar", sorted(blocked))
                candidates = {p: t for p, t in candidates.items() if p not in blocked}
            elif not basis_report.source_ok:
                log.warning("basis check unavailable (%s); proceeding on venue data alone", basis_report.error)

        selection = self.selector.select(
            candidates,
            required_notional=self.cfg.depth_target_notional,
            can_trade=can_trade,
            previous=self.universe or self.book.positions.keys(),
        )
        self.selection = selection
        self.universe = selection.selected
        self.journal.universe(now_ms, selection.to_dict())
        if not self.universe:
            log.warning("Rule 1 produced an empty universe this bar")

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def _context(self, now_ms: int, view: Optional[PortfolioView] = None) -> MarketContext:
        positions = (view.positions if view else None) or self.book.held()
        cash_usd = view.cash_usd if view else self._cash_usd(self.balances)
        nav = view.nav if view else portfolio_nav(cash_usd, positions)
        current_bar = bar_index(now_ms, self.cfg.bar_seconds)
        blocked = self.risk.blocked_pairs(current_bar)
        # Hand the strategy only the context it declares it needs. Passing the
        # whole rolling buffer is correct but makes every indicator re-scan
        # hundreds of bars per pair per bar.
        context_bars = max(1, self.strategy.max_context_bars)
        candles = {
            pair: self.builder.history(pair)[-context_bars:] for pair in self._tracked_pairs()
        }
        return MarketContext(
            now_ms=now_ms,
            bar_seconds=self.cfg.bar_seconds,
            candles=candles,
            tickers=self.tickers,
            nav=nav,
            cash_usd=cash_usd,
            positions=positions,
            universe=list(self.universe),
            blocked=blocked,
            daily_halt=self.risk.daily_halt,
            bar_index=current_bar,
            state={},
        )

    def _tracked_pairs(self) -> list[str]:
        pairs = set(self.builder.histories()) | set(self.universe) | set(self.book.positions)
        return sorted(pairs)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def _feed_candles(self, tickers: dict[str, Ticker], now_ms: int) -> None:
        for pair, ticker in tickers.items():
            if pair not in self.exchange_pairs:
                continue
            self.builder.add(
                pair=pair,
                ts_ms=now_ms,
                price=ticker.mid,
                bid=ticker.max_bid,
                ask=ticker.min_ask,
                cumulative_volume=ticker.unit_volume,
            )

    def _safe_short_positions(self) -> Optional[list[Any]]:
        """The venue's open shorts, or ``None`` when the question went unanswered.

        ``None`` and ``[]`` are different answers and the caller must not conflate
        them. ``[]`` means "the venue holds no shorts"; returning ``[]`` for a
        transport failure made the pruning pass below delete every short in the
        book -- cost basis, collateral, the Rule 5 stop and the Rule 6 open time,
        none of which the API can report back.
        """
        try:
            return self.client.short_positions()
        except Exception as exc:
            # Shorts may be disabled for the competition; that must not be fatal.
            log.warning("short_positions unavailable (%s); keeping the book's shorts this cycle", exc)
            return None

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def _reconcile_positions(
        self, balances: dict[str, WalletBalance], shorts: Optional[list[Any]]
    ) -> None:
        """Make the local book agree with the exchange.

        The exchange is authoritative for *quantity* (it is the thing that
        settles trades). The local book is authoritative for *cost basis and
        stop levels*, which the API never reports. A disagreement means a fill we
        did not see -- an unknown-outcome order, a missed cycle, or a manual
        intervention -- and is journalled rather than silently absorbed.

        A row that is simply *absent* from the payload is not evidence of a flat
        position. A truncated balance response would otherwise read every missing
        row as "the exchange holds nothing" and delete the entire book -- the
        failure this guard exists to prevent. Only an explicit row showing a zero
        balance closes a position.
        """
        for pair, trade_pair in self.exchange_pairs.items():
            balance = balances.get(trade_pair.coin)
            exchange_qty = balance.total if balance else 0.0
            position = self.book.get(pair)
            local_qty = position.quantity if position and not position.is_short else 0.0
            # Half a lot: a smaller gap is rounding, not a real disagreement.
            tolerance = max(1e-9, 0.5 * 10.0 ** (-trade_pair.amount_precision))

            if abs(exchange_qty - local_qty) <= tolerance:
                continue
            self.stats.reconciliations += 1
            if exchange_qty <= tolerance:
                if balance is None:
                    # No row for this coin at all: the snapshot is incomplete, not
                    # a flat balance. Keep the position and say so loudly.
                    log.warning(
                        "%s: no %s row in the balance payload; keeping the local position "
                        "(local=%.10f) rather than treating an absent row as a close",
                        pair,
                        trade_pair.coin,
                        local_qty,
                    )
                    self.journal.reconciliation(
                        int(time.time() * 1000),
                        pair,
                        "row_missing_kept",
                        {"local": local_qty, "coin": trade_pair.coin},
                    )
                    continue
                if position is not None and not position.is_short:
                    self.book.positions.pop(pair, None)
                self.journal.reconciliation(
                    int(time.time() * 1000),
                    pair,
                    "closed_externally",
                    {"local": local_qty, "exchange": exchange_qty},
                )
                continue
            if position is None or position.is_short:
                # Adopt an unknown holding. The cost basis is unknown, but the stop
                # usually is not: if we sent an order for this pair and never
                # learned its outcome, the remembered intent holds the level the
                # entry was approved with. Adopting without it leaves the position
                # with no Rule 5 stop at all -- and with the time stop disabled
                # there is nothing else that would ever close it.
                ticker = self.tickers.get(pair)
                intent = self._unknown_intents.get(pair)
                remembered = (intent or {}).get("stop_price")
                stop_price = (
                    float(remembered)
                    if isinstance(remembered, (int, float)) and remembered > 0
                    else None
                )
                self._forget_intent(pair, int(time.time() * 1000), "adopted from the exchange")
                self.book.apply_spot_buy(
                    pair,
                    exchange_qty,
                    ticker.mid if ticker is not None else 0.0,
                    int(time.time() * 1000),
                    stop_price,
                )
                self.journal.reconciliation(
                    int(time.time() * 1000),
                    pair,
                    "adopted_unknown_holding",
                    {
                        "local": local_qty,
                        "exchange": exchange_qty,
                        "stop_restored": stop_price is not None,
                    },
                )
                if stop_price is None:
                    log.error(
                        "%s: adopted %.10f from the exchange with no known cost basis and no "
                        "remembered stop; it is unprotected until another exit closes it",
                        pair,
                        exchange_qty,
                    )
                else:
                    log.warning(
                        "%s: adopted %.10f from the exchange and restored the stop at %.8f",
                        pair,
                        exchange_qty,
                        stop_price,
                    )
            else:
                position.quantity = exchange_qty
                self.journal.reconciliation(
                    int(time.time() * 1000),
                    pair,
                    "quantity_corrected",
                    {"local": local_qty, "exchange": exchange_qty},
                )

        if shorts is None:
            # The venue did not answer, so we cannot tell "no shorts" from "we do
            # not know". Pruning on an unanswered question deletes live positions.
            return
        short_pairs = {sp.pair for sp in shorts}
        for pair, position in list(self.book.positions.items()):
            if position.is_short and pair not in short_pairs:
                self.book.positions.pop(pair, None)
                self.journal.reconciliation(int(time.time() * 1000), pair, "short_closed_externally")
        for short in shorts:
            self._adopt_unknown_short(short)

    def _adopt_unknown_short(self, short: Any) -> None:
        """Adopt a venue-held short, keeping the real open time and the local stop."""
        pair = getattr(short, "pair", "")
        quantity = float(getattr(short, "quantity", 0.0) or 0.0)
        if not pair or not math.isfinite(quantity) or quantity <= 0:
            return

        now_ms = int(time.time() * 1000)
        entry = float(getattr(short, "entry_price", 0.0) or 0.0)
        collateral = float(getattr(short, "collateral", 0.0) or 0.0)

        try:
            created = int(getattr(short, "created_ts_ms", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            created = 0

        valid_created = 0 < created <= now_ms
        opened_ts_ms = created if valid_created else now_ms

        position = self.book.get(pair)

        # Already short: sync the venue's fields, keep the local stop.
        if position is not None and position.is_short:
            before = {
                "quantity": position.quantity,
                "entry": position.avg_price,
                "collateral": position.collateral,
                "opened_ts_ms": position.opened_ts_ms,
            }

            position.quantity = quantity

            if math.isfinite(entry) and entry > 0:
                position.avg_price = entry
            if math.isfinite(collateral) and collateral >= 0:
                position.collateral = collateral
            if valid_created:
                position.opened_ts_ms = created

            after = {
                "quantity": position.quantity,
                "entry": position.avg_price,
                "collateral": position.collateral,
                "opened_ts_ms": position.opened_ts_ms,
            }

            if before != after:
                self.journal.reconciliation(
                    now_ms,
                    pair,
                    "short_state_corrected",
                    {"local": before, "exchange": after},
                )
            return

        # The book already holds a long for this pair; a short must not overwrite it.
        if position is not None:
            log.error(
                "%s: venue reports a short while the book holds a long",
                pair,
            )
            self.journal.reconciliation(
                now_ms, pair, "short_conflicts_with_long"
            )
            return

        # Newly adopted short: take the venue's open time.
        if not math.isfinite(entry) or entry <= 0:
            ticker = self.tickers.get(pair)
            entry = ticker.mid if ticker is not None else 0.0

        if not math.isfinite(collateral) or collateral < 0:
            log.error("%s: invalid short collateral; cannot adopt", pair)
            return

        # A short created by an order whose outcome we never learned keeps the
        # stop that entry was approved with, for the same reason as the long path:
        # without it the position has no Rule 5 level at all.
        intent = self._unknown_intents.get(pair)
        remembered = (intent or {}).get("stop_price")
        stop_price = (
            float(remembered)
            if isinstance(remembered, (int, float)) and remembered > 0
            else None
        )
        self._forget_intent(pair, now_ms, "adopted short from the exchange")

        self.book.apply_short_open(
            pair,
            quantity,
            entry,
            collateral,
            opened_ts_ms,
            stop_price,
        )

        self.journal.reconciliation(
            now_ms,
            pair,
            "adopted_unknown_short",
            {
                "quantity": quantity,
                "entry": entry,
                "collateral": collateral,
                "opened_ts_ms": opened_ts_ms,
                "stop_restored": stop_price is not None,
            },
        )

        if stop_price is None:
            log.error(
                "%s: adopted short %.10f @ %.8f with no known cost basis and no remembered "
                "stop; it is unprotected until another exit closes it",
                pair,
                quantity,
                entry,
            )
        else:
            log.warning(
                "%s: adopted short %.10f @ %.8f and restored the stop at %.8f",
                pair,
                quantity,
                entry,
                stop_price,
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute(self, actions: list[ApprovedAction], now_ms: int, current_bar: int) -> None:
        for action in actions:
            try:
                if action.action in (ENTER_LONG, ENTER_SHORT):
                    # Persist the intent BEFORE the request leaves. The crash window
                    # is between sending and hearing back, and `_persist` only runs
                    # at the end of the cycle -- so an intent recorded after the
                    # fact is an intent a crash can lose, and the position it
                    # created would then be adopted with no stop.
                    if not self._write_intent(action, now_ms):
                        self.stats.order_errors += 1
                        continue
                if action.action == ENTER_LONG:
                    self._enter_long(action, now_ms, current_bar)
                elif action.action == ENTER_SHORT:
                    self._enter_short(action, now_ms, current_bar)
                elif action.action == EXIT_LONG:
                    self._exit_long(action, now_ms, current_bar)
                elif action.action == EXIT_SHORT:
                    self._exit_short(action, now_ms, current_bar)
            except Exception as exc:
                self.stats.order_errors += 1
                log.exception("execution failed for %s", action.pair)
                self.journal.error("execute", str(exc), ts_ms=now_ms, action=action.to_dict())

    def _enter_long(self, action: ApprovedAction, now_ms: int, current_bar: int) -> None:
        trade_pair = self.exchange_pairs.get(action.pair)
        ticker = self.tickers.get(action.pair)
        if trade_pair is None or ticker is None:
            return

        if self.cfg.limit_entries:
            self._enter_long_passive(action, trade_pair, ticker, now_ms, current_bar)
            return

        quantity = self._quantise(trade_pair, action.quantity, ticker.mid)
        if quantity is None:
            self.journal.order(now_ms, action.to_dict(), error="below pair minimum")
            return

        self.stats.orders_sent += 1
        result = self.client.place_order(action.pair, "BUY", trade_pair.round_qty(quantity), "MARKET")
        self.journal.order(now_ms, action.to_dict(), result=result)
        self._record_result(result, action, now_ms, current_bar)

    def _limit_price(self, pair: str, trade_pair: TradePair, ticker: Ticker) -> float:
        """The passive bid: a touch below the mid, rounded to the pair's tick."""
        offset = self.cfg.limit_entry_offset_bps / 10_000.0
        return float(fmt(ticker.mid * (1.0 - offset), trade_pair.price_precision))

    def _enter_long_passive(
        self, action: ApprovedAction, trade_pair: TradePair, ticker: Ticker, now_ms: int, current_bar: int
    ) -> None:
        """Post a resting maker bid instead of crossing the spread.

        Sized from the *limit* price rather than the mid, so the notional the risk
        layer approved is the notional we end up holding -- a fill below the mid at
        mid-based sizing would quietly exceed the cap.

        The order is tracked in ``self._resting`` purely so it can be cancelled if
        it goes stale. `_refresh_pending` already asks the venue for resting
        orders, so the risk layer reserves the capital for this pair on the next
        bar and will not size a second entry on top of it.
        """
        if action.pair in self._resting:
            # One resting bid per pair is enough; the risk layer agrees.
            return
        price = self._limit_price(action.pair, trade_pair, ticker)
        if price <= 0:
            return
        quantity = self._quantise(trade_pair, action.quantity, price)
        if quantity is None:
            self.journal.order(now_ms, action.to_dict(), error="below pair minimum at the limit price")
            return

        self.stats.orders_sent += 1
        self.stats.orders_posted += 1
        result = self.client.place_order(
            action.pair, "BUY", trade_pair.round_qty(quantity), "LIMIT", price=price
        )
        self.journal.order(now_ms, action.to_dict(), result=result)
        if result.status in ("FILLED", "PENDING"):
            self._resting[action.pair] = (current_bar, price)
        if result.status == "PENDING":
            log.info("%s: resting bid %s for %s (maker)", action.pair, price, quantity)
            return
        self._record_result(result, action, now_ms, current_bar)

    def _cancel_all_entry_orders(self, now_ms: int) -> None:
        """Cancel every entry order: the local table first, then the venue's own.

        ``_expire_resting_orders`` only touches orders older than
        ``LIMIT_ENTRY_TIMEOUT_BARS``, which is right during normal trading and wrong
        on a halt: an order that is still fresh can still fill, and it must not fill
        into an account nothing is watching any more.

        The venue is asked as well, because ``_resting`` does not survive a restart.
        An order only the venue knows about is still an order -- that is exactly the
        leftover a restart leaves behind -- so cancelling by pair is the only way to
        reach it. A cancel that fails leaves the order in play, and the halt stays
        alive until the account is genuinely clean.
        """
        pairs = set(self._resting)
        try:
            total, by_pair = self.client.pending_count()
        except Exception as exc:
            # Do not pretend the list is empty: the local table is not the whole
            # truth, so an unanswered query means "may still be orders".
            log.error("could not list venue orders while halting (%s); will retry", exc)
        else:
            pairs |= {p for p, count in by_pair.items() if count and count > 0}

        for pair in sorted(pairs):
            try:
                self.client.cancel_order(pair=pair)
            except Exception as exc:
                # Leave it tracked and try again next cycle. A cancel that failed
                # must not be assumed to have happened.
                log.warning("%s: could not cancel the resting bid while halting: %s", pair, exc)
                self.journal.error("cancel", str(exc), ts_ms=now_ms, pair=pair)
                continue
            self._resting.pop(pair, None)
            self.journal.event("cancel", pair=pair, reason="halt")

    def _halt_outstanding(self) -> str:
        """What still prevents a halted engine from being finished ('' if nothing).

        Deliberately reports an unanswered query as outstanding: assuming the venue
        holds nothing is exactly how an order gets orphaned.
        """
        parts: list[str] = []
        held = sorted(self.book.held())
        if held:
            parts.append(f"{len(held)} position(s) {held}")
        if self._resting:
            parts.append(f"{len(self._resting)} tracked order(s) {sorted(self._resting)}")
        if self._unknown_intents:
            parts.append(
                f"{len(self._unknown_intents)} unresolved intent(s) {sorted(self._unknown_intents)}"
            )
        try:
            total, by_pair = self.client.pending_count()
        except Exception as exc:
            parts.append(f"pending-count query failed ({exc})")
        else:
            if total:
                parts.append(f"{total} venue order(s) {sorted(by_pair)}")
        return "; ".join(parts)

    def _expire_resting_orders(self, now_ms: int, current_bar: int) -> None:
        """Cancel maker entries that have gone stale, so capital is not locked up.

        A resting order is capital committed and invisible: if the reversion has
        already happened by the time it would fill, the entry is no longer the one
        the strategy asked for. Cancelling is what bounds that exposure -- without
        it, a bid below the market can sit there through the whole move.
        """
        if not self._resting:
            return
        timeout = max(1, int(self.cfg.limit_entry_timeout_bars))
        for pair, (placed_bar, price) in list(self._resting.items()):
            if current_bar - placed_bar < timeout:
                continue
            try:
                cancelled = self.client.cancel_order(pair=pair)
            except Exception as exc:
                # Leave it in the book and try again next loop; a cancel that
                # failed must not be assumed to have happened.
                log.warning("%s: could not cancel the stale resting bid: %s", pair, exc)
                self.journal.error("cancel", str(exc), ts_ms=now_ms, pair=pair, price=price)
                continue
            self._resting.pop(pair, None)
            self.stats.orders_cancelled += 1
            log.info("%s: cancelled the stale resting bid at %s (%s)", pair, price, cancelled)
            self.journal.event(
                "order_cancelled", ts_ms=now_ms, pair=pair, price=price, cancelled=cancelled, reason="stale entry"
            )

    def _enter_short(self, action: ApprovedAction, now_ms: int, current_bar: int) -> None:
        collateral = max(1.0, round(action.collateral, 2))
        if collateral < 1.0:
            return
        self.stats.orders_sent += 1
        try:
            payload = self.client.short_open(action.pair, collateral)
        except Exception as exc:
            # A transport failure leaves the short's existence unknown, exactly as
            # it does for a spot order. It cannot be re-sent (that risks a second
            # position), so remember the intent -- pair, size and the stop the
            # entry was approved with -- and let `_reconcile_positions` adopt
            # whatever the venue turns out to hold, with its stop restored.
            self.stats.unknown_orders += 1
            # The intent is already durable: `_execute` persisted it before calling
            # this, so a crash from here still leaves the stop on disk.
            log.error("short_open %s transport failure (state UNKNOWN): %s", action.pair, exc)
            self.journal.error("execute", f"short_open transport failure: {exc}", ts_ms=now_ms, pair=action.pair)
            return
        self.journal.order(now_ms, action.to_dict(), result=payload)
        if not is_success(payload):
            self.stats.order_errors += 1
            log.warning("short_open %s rejected: %s", action.pair, payload.get("ErrMsg"))
            return
        quantity = float(payload.get("ShortQty", 0.0) or 0.0)
        entry = float(payload.get("EntryPrice", 0.0) or 0.0)
        if payload.get("Status") != "OPEN" or quantity <= 0:
            # A resting LIMIT order has no position yet.
            return
        self.book.apply_short_open(
            action.pair, quantity, entry, float(payload.get("Collateral", collateral) or collateral),
            now_ms, action.stop_price,
        )
        self.stats.entries += 1
        self.journal.trade(
            now_ms, action.pair, action.action, "SHORT_OPEN", quantity, entry,
            float(payload.get("OpenFee", 0.0) or 0.0), payload.get("ID", ""), "SHORT", action.reason,
        )

    def _exit_long(self, action: ApprovedAction, now_ms: int, current_bar: int) -> None:
        trade_pair = self.exchange_pairs.get(action.pair)
        position = self.book.get(action.pair)
        if trade_pair is None or position is None:
            return
        ticker = self.tickers.get(action.pair)
        # A held position must always be closable. `mark_price` is only a *local*
        # estimate -- it can be zero for an adopted holding or a position restored
        # from a bad state file -- so fall back to the live quote and then to the
        # entry price. Refusing to sell because the reference price is unusable
        # turned a bad mark into a position that nothing, not even the kill
        # switch, could exit.
        reference = position.mark_price if position.mark_price > 0 else 0.0
        if reference <= 0 and ticker is not None and ticker.mid > 0:
            reference = ticker.mid
        if reference <= 0 and position.avg_price > 0:
            reference = position.avg_price
        if reference <= 0:
            # Nothing to price the minimum against; sell the whole holding rather
            # than leaving it stranded.
            quantity = float(fmt(position.quantity, trade_pair.amount_precision))
            if quantity <= 0:
                self.journal.order(now_ms, action.to_dict(), error="position quantity rounds to zero")
                return
            log.warning(
                "%s: no usable price for the minimum-order check; selling the full balance %.10f",
                action.pair,
                quantity,
            )
        else:
            quantity = self._quantise(trade_pair, position.quantity, reference)
            if quantity is None:
                self.journal.order(now_ms, action.to_dict(), error="nothing sellable above the pair minimum")
                return
            # A SELL that would leave dust behind is rounded to the full balance.
            if quantity < position.quantity * 0.999:
                quantity = position.quantity
                quantity = float(fmt(quantity, trade_pair.amount_precision))

        self.stats.orders_sent += 1
        result = self.client.place_order(action.pair, "SELL", trade_pair.round_qty(quantity), "MARKET")
        self.journal.order(now_ms, action.to_dict(), result=result)
        self._record_result(result, action, now_ms, current_bar)

    def _exit_short(self, action: ApprovedAction, now_ms: int, current_bar: int) -> None:
        self.stats.orders_sent += 1
        try:
            payload = self.client.short_close(action.pair)
        except Exception as exc:
            # Deliberately NOT remembered as an unknown intent. The short is still
            # in the book, so it already holds its slot and its stop, and the next
            # cycle's protective exits will generate another close. Recording an
            # intent here would also be wrong: `_resolve_unknown_intents` re-queries
            # the *spot* order history, where a short close does not appear and an
            # unrelated row could match.
            self.stats.unknown_orders += 1
            log.error("short_close %s transport failure (state UNKNOWN): %s", action.pair, exc)
            self.journal.error("execute", f"short_close transport failure: {exc}", ts_ms=now_ms, pair=action.pair)
            return
        self.journal.order(now_ms, action.to_dict(), result=payload)
        if not is_success(payload):
            self.stats.order_errors += 1
            log.warning("short_close %s rejected: %s", action.pair, payload.get("ErrMsg"))
            return
        closed = float(payload.get("ClosedQty", 0.0) or 0.0)
        remaining = payload.get("RemainingQty")
        price = float(payload.get("ClosePrice", 0.0) or 0.0)
        if closed <= 0:
            # A success payload with nothing closed: do not count an exit or start
            # a Rule 12 cooldown for a position that is still open.
            self.stats.order_errors += 1
            log.warning(
                "short_close %s reported Success but ClosedQty=%r (FullyClosed=%r); position left open",
                action.pair,
                payload.get("ClosedQty"),
                payload.get("FullyClosed"),
            )
            return
        self.book.apply_short_close(action.pair, closed, price)
        self.stats.exits += 1
        self.risk.record_exit(action.pair, current_bar)
        self.journal.trade(
            now_ms, action.pair, action.action, "SHORT_CLOSE", closed, price,
            float(payload.get("CloseFee", 0.0) or 0.0), "", "SHORT", action.reason,
        )

    def _record_result(self, result: OrderResult, action: ApprovedAction, now_ms: int, current_bar: int) -> None:
        """Apply a spot order outcome to the book, reconciling UNKNOWNs."""
        if result.status == "UNKNOWN":
            self.stats.unknown_orders += 1
            outcome, reconciled = self._reconcile_unknown_order(action, result, now_ms)
            if outcome == ORDER_NOT_FILLED:
                self._forget_intent(action.pair, now_ms, "the venue reports it did not fill")
                return
            if outcome == ORDER_UNKNOWN:
                # The intent is already durable: `_execute` wrote it *before*
                # sending, which is the only way to cover a crash between the
                # request leaving and the reply arriving. It stays on disk and is
                # re-queried every cycle.
                log.error(
                    "%s: outcome still unknown; the intent is on disk and retried each cycle",
                    action.pair,
                )
                return
            result = reconciled

        if result.status == "REJECTED":
            self.stats.order_errors += 1
            log.warning("%s %s rejected: %s", result.side, action.pair, result.err_msg)
            return
        if result.status != "FILLED":
            # Resting limit order: no position change yet.
            return

        # A FILLED status with no filled quantity is a contradiction -- the venue
        # told us the order went through but reported zero units. Booking
        # `action.quantity` instead (the old behaviour) invents a position that
        # may never have existed, at a price that may be zero. Refuse it and let
        # the next cycle's balance reconciliation settle the truth.
        if not (result.filled_quantity > 0):
            self.stats.order_errors += 1
            log.error(
                "%s %s reported FILLED with filled_quantity=%r; not booking a position "
                "(reconciliation will settle it)",
                result.side,
                action.pair,
                result.filled_quantity,
            )
            self.journal.error(
                "execute",
                "FILLED with no filled quantity; refusing to book",
                ts_ms=now_ms,
                pair=action.pair,
                status=result.status,
                filled=result.filled_quantity,
            )
            return

        filled = result.filled_quantity
        if not (result.avg_fill_price > 0):
            self.stats.order_errors += 1
            log.error(
                "%s %s filled with no usable average price (%r); not booking a position",
                result.side,
                action.pair,
                result.avg_fill_price,
            )
            return
        price = result.avg_fill_price or (action.price or 0.0)
        if result.side == "BUY":
            self.book.apply_spot_buy(action.pair, filled, price, now_ms, action.stop_price)
            self.stats.entries += 1
        else:
            self.book.apply_spot_sell(action.pair, filled, price)
            self.stats.exits += 1
            self.risk.record_exit(action.pair, current_bar)
        self.journal.trade(
            now_ms, action.pair, action.action, result.side, filled, price,
            result.commission, result.order_id, result.role, action.reason,
        )

    def _resolve_unknown_intents(self, now_ms: int, current_bar: int) -> None:
        """Re-ask the venue about every order whose outcome is still unknown.

        Runs every cycle. Until one resolves, its capital and its slot stay
        reserved (see `_refresh_pending`), so the risk layer cannot spend the same
        money twice on an order that may already have filled.
        """
        for pair, intent in list(self._unknown_intents.items()):
            if pair not in self.exchange_pairs:
                continue
            action = ApprovedAction(
                pair=pair,
                action=str(intent.get("action") or ENTER_LONG),
                quantity=float(intent.get("quantity") or 0.0),
                notional=float(intent.get("notional") or 0.0),
                reason="retry of an unresolved order",
                stop_price=intent.get("stop_price"),
            )
            probe = OrderResult(
                pair=pair,
                side=str(intent.get("side") or "BUY"),
                order_type="MARKET",
                quantity=action.quantity,
                price=0.0,
                status="UNKNOWN",
                err_msg="retry",
            )
            intent["attempts"] = int(intent.get("attempts") or 0) + 1
            outcome, resolved = self._reconcile_unknown_order(
                action, probe, now_ms, sent_ms=int(intent.get("sent_ms") or now_ms)
            )
            if outcome == ORDER_UNKNOWN:
                age_ms = now_ms - int(intent.get("sent_ms") or now_ms)
                # An intent that can never be settled still holds its reserve, so make
                # that visible instead of letting it sit there indefinitely. It is not
                # released on a guess: only a terminal venue status does that.
                if age_ms > 1_800_000 and intent["attempts"] % 10 == 1:
                    log.error(
                        "%s: order intent unresolved for %.0f minutes after %d attempts; it is "
                        "still holding its reserve. Check the venue by hand.",
                        pair,
                        age_ms / 60_000.0,
                        intent["attempts"],
                    )
                continue
            self._forget_intent(pair, now_ms, f"resolved: {outcome}")
            if outcome == ORDER_NOT_FILLED:
                continue
            log.warning(
                "%s: unresolved order settled after %d attempt(s)", pair, intent["attempts"]
            )
            # `action` carries the remembered stop, so a fill booked here gets the
            # protective level the original order was approved with.
            self._record_result(resolved, action, now_ms, current_bar)

    def _reconcile_unknown_order(
        self, action: ApprovedAction, result: OrderResult, now_ms: int, sent_ms: Optional[int] = None
    ) -> tuple[str, Optional[OrderResult]]:
        """Did an UNKNOWN order actually land?

        Never re-send. Ask the order history instead: if an order for this pair
        was created after we sent the request and matches the side, the quantity
        and is genuinely filled, treat it as ours; otherwise treat it as not
        placed.

        Every clause here is load-bearing. Matching on side and recency alone
        accepted a *cancelled* order of an unrelated size as proof that our order
        went through -- so the position was never booked and the intent was
        silently dropped. And because ``OrderResult.from_api`` reads whatever the
        row happens to contain, a row with no ``Status`` used to arrive as
        ``FILLED`` with ``FilledQuantity=0``, which the execution path then booked
        as a full-size position at price zero.
        """
        try:
            rows = self.client.query_orders(pair=action.pair, limit=20)
        except Exception as exc:
            self.journal.reconciliation(now_ms, action.pair, "query_failed", {"error": str(exc)})
            log.error("cannot reconcile unknown order for %s: %s", action.pair, exc)
            return (ORDER_UNKNOWN, None)

        trade_pair = self.exchange_pairs.get(action.pair)
        # One lot, with the same half-lot slack the balance reconciliation uses.
        lot = 10.0 ** (-trade_pair.amount_precision) if trade_pair is not None else 1e-9
        tolerance = max(1e-9, 0.5 * lot)

        # Anchored on when the order was *sent*, not on now: a retry that runs
        # minutes later must still accept the row the original request created.
        cutoff = (sent_ms if sent_ms is not None else now_ms) - 120_000
        expected_side = "BUY" if action.action == ENTER_LONG else "SELL"
        alive = False
        for row in rows:
            created = int(row.get("CreateTimestamp", 0) or 0)
            side = str(row.get("Side", "")).upper()
            status = str(row.get("Status", "") or "").upper()
            quantity = float(row.get("Quantity", 0.0) or 0.0)
            filled = float(row.get("FilledQuantity", 0.0) or 0.0)
            if created < cutoff or side != expected_side:
                continue
            if quantity <= 0 or abs(quantity - action.quantity) > tolerance:
                continue
            if status == "FILLED":
                if filled <= 0:
                    continue
                self.journal.reconciliation(
                    now_ms,
                    action.pair,
                    "order_found",
                    {"order_id": row.get("OrderID"), "status": status, "filled": filled},
                )
                return (
                    ORDER_FILLED,
                    OrderResult.from_api(
                        action.pair,
                        expected_side,
                        "MARKET",
                        action.quantity,
                        {"Success": True, "OrderDetail": row},
                    ),
                )
            if status in TERMINAL_NOT_FILLED:
                # A terminal state that is not a fill settles the question: the order
                # is gone and no position will appear from it, so the intent and its
                # reserve can be released. Holding it forever is its own failure mode,
                # because an intent that can never settle also blocks a halt from
                # ever finishing.
                self.journal.reconciliation(
                    now_ms, action.pair, "order_terminal_unfilled", {"status": status}
                )
                log.warning(
                    "%s: the unknown order is %s at the venue; releasing its intent",
                    action.pair,
                    status,
                )
                return (ORDER_NOT_FILLED, None)
            # PENDING, NEW, PARTIALLY_FILLED: the order is still alive, so the outcome
            # is still unknown and the reserve has to stay.
            alive = True
        self.journal.reconciliation(
            now_ms,
            action.pair,
            "order_alive" if alive else "order_not_found",
            {"reason": result.err_msg},
        )
        # "Not found" is not "not filled": a row can be missing because the venue has
        # not written it yet. Only a terminal status releases the reserve.
        return (ORDER_UNKNOWN, None)

    def _flatten(self, now_ms: int, reason: str) -> None:
        actions = self.risk.flatten_all(self.book.held())
        if not actions:
            return
        log.warning("flattening %d position(s): %s", len(actions), reason)
        self._execute(actions, now_ms, bar_index(now_ms, self.cfg.bar_seconds))

    @staticmethod
    def _quantise(trade_pair: TradePair, quantity: float, price: float) -> Optional[float]:
        """Round down to the pair's lot size, rejecting dust."""
        if quantity <= 0 or price <= 0:
            return None
        rounded = float(fmt(quantity, trade_pair.amount_precision))
        if rounded <= 0:
            return None
        if rounded * price < trade_pair.min_order:
            return None
        return rounded

    @staticmethod
    def _balances_usable(balances: dict[str, WalletBalance]) -> bool:
        """Is this snapshot complete enough to price the book and reconcile it?

        The quote currency must be present. An absent ``USD`` row means either a
        partial response or a venue that renamed its quote asset; either way the
        numbers derived from it are fiction, so the cycle is skipped.

        This is necessary but not sufficient -- a payload can carry USD and
        nothing else, which prices the portfolio correctly while saying nothing
        about the coins we hold. ``_reconcile_positions`` therefore refuses to
        close a position on a *missing* row and only acts on an explicit zero.
        """
        return "USD" in balances

    @staticmethod
    def _cash_usd(balances: dict[str, WalletBalance]) -> float:
        usd = balances.get("USD")
        return usd.total if usd else 0.0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _state_path(self) -> Path:
        # Per mode, for the same reason as the position book: a stamp stops a file
        # being *read* across runtimes, a separate file stops it being *written*
        # over. See Config.state_dir_name.
        return self._state_dir() / "engine_state.json"

    def _scope_matches(self, saved: dict[str, Any]) -> bool:
        """Does a state file's stamp match this process? See Config.state_scope."""
        current = self.cfg.state_scope
        return all(saved.get(key) == current[key] for key in current)

    def _state_dir(self) -> Path:
        """Where this runtime's state lives: a per-mode subdirectory of JOURNAL_DIR."""
        return Path(self.cfg.journal_dir) / self.cfg.state_dir_name

    def _intent_dir(self) -> Path:
        """One small file per unresolved order, written *before* it is sent."""
        return self._state_dir() / "intents"

    def _intent_path(self, pair: str) -> Path:
        return self._intent_dir() / f"{pair.replace('/', '-')}.json"

    def _migrate_legacy_state(self) -> None:
        """Adopt state written before it lived in a per-mode directory.

        State used to sit directly in JOURNAL_DIR. Moving it into
        ``<journal_dir>/<mode>/`` -- so one runtime can never overwrite another's --
        would otherwise make an upgrade look like a first run: the book would be
        rebuilt by balance reconciliation with no cost basis and no stop, and a
        persisted halt would be dropped silently. The halt is the one piece of state
        that must never disappear without saying so.

        A legacy file is adopted only when its provenance is absent (it predates
        stamping) or matches this runtime. A file naming a different runtime is left
        untouched and reported: it belongs to a deployment this process is not, and
        guessing would be worse than refusing.
        """
        import json
        import shutil

        legacy_dir = Path(self.cfg.journal_dir)
        target_dir = self._state_dir()
        for name in ("positions.json", "engine_state.json"):
            source = legacy_dir / name
            target = target_dir / name
            if not source.is_file() or target.exists():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                log.error("legacy state %s is unreadable (%s); leaving it in place", source, exc)
                self.journal.error("migrate", f"unreadable legacy state: {exc}", path=str(source))
                continue
            saved = payload.get("scope") if isinstance(payload, dict) else None
            if isinstance(saved, dict) and not self._scope_matches(saved):
                log.error(
                    "legacy state %s belongs to %s@%s, not %s@%s; refusing to migrate it. "
                    "Move it aside by hand if this deployment really is its owner.",
                    source,
                    saved.get("mode"),
                    saved.get("venue"),
                    self.cfg.state_scope["mode"],
                    self.cfg.state_scope["venue"],
                )
                self.journal.error(
                    "migrate", "legacy state belongs to another runtime", path=str(source)
                )
                continue
            if saved is None and self.cfg.mock:
                # Unstamped, so it could belong to either runtime -- and migrating is
                # destructive (the original is moved). A simulator run must not claim
                # state it may not own: smoke-testing in mock first would otherwise
                # take the live book with it, and the next live start would find
                # nothing. Leave it for the live run, or for a human.
                log.warning(
                    "legacy state %s has no provenance and this is a mock run; leaving it "
                    "alone. Start live to migrate it, or move it by hand.",
                    source,
                )
                continue
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                # Copy first so a crash mid-migration cannot lose the original.
                shutil.copy2(source, source.with_name(source.name + ".pre-migration"))
                source.replace(target)
            except Exception as exc:
                log.error("could not migrate %s (%s); it is untouched", source, exc)
                continue
            log.warning(
                "migrated legacy state %s -> %s (backup kept as *.pre-migration)", source, target
            )
            self.journal.event("state_migrated", source=str(source), target=str(target))

    def _load_unknown_intents(self) -> None:
        """Restore order intents that were written before their orders were sent.

        This is what closes the crash window: the intent is on disk *before* the
        request leaves, so a process that dies between sending and hearing back
        still knows what it sent and what stop it belonged with.
        """
        import json

        directory = self._intent_dir()
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.error("unreadable order intent %s (%s); leaving it in place", path, exc)
                continue
            pair = str(payload.get("pair") or "")
            if not pair:
                continue
            self._unknown_intents[pair] = payload
        if self._unknown_intents:
            log.error(
                "restored %d order intent(s) written before sending (%s); re-queried every cycle",
                len(self._unknown_intents),
                ", ".join(sorted(self._unknown_intents)),
            )

    def _write_intent(self, action: ApprovedAction, now_ms: int) -> bool:
        """Persist an order intent *before* the order is sent. False means do not send.

        An order the process cannot remember is an order whose fill may arrive with
        no stop attached: the balance reconciliation would adopt the position and
        nothing would know where the protective level belonged.
        """
        import json

        pair = action.pair
        previous = self._unknown_intents.get(pair) or {}
        intent = {
            "pair": pair,
            "action": action.action,
            "side": "BUY" if action.action == ENTER_LONG else "SELL",
            "quantity": float(action.quantity or 0.0),
            "notional": float(action.notional or 0.0),
            "stop_price": float(action.stop_price) if (action.stop_price and action.stop_price > 0) else None,
            "sent_ms": int(previous.get("sent_ms") or now_ms),
            "attempts": int(previous.get("attempts") or 0),
        }
        path = self._intent_path(pair)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(intent, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            log.error("could not persist the order intent for %s (%s); NOT sending", pair, exc)
            self.journal.error("intent", f"could not persist intent: {exc}", ts_ms=now_ms, pair=pair)
            return False
        self._unknown_intents[pair] = intent
        return True

    def _forget_intent(self, pair: str, now_ms: int, why: str) -> None:
        """Drop a pair's intent, in memory and on disk, once its order is settled."""
        self._unknown_intents.pop(pair, None)
        try:
            self._intent_path(pair).unlink(missing_ok=True)
        except Exception as exc:
            log.error("could not remove the order intent for %s (%s)", pair, exc)
        self.journal.event("intent_cleared", pair=pair, reason=why, ts_ms=now_ms)

    def _load_risk_state(self) -> None:
        path = self._state_path()
        if not path.is_file():
            return
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"state file is a {type(payload).__name__}, not an object")
            saved_scope = payload.get("scope")
            if isinstance(saved_scope, dict) and not self._scope_matches(saved_scope):
                # Another runtime's state: a mock file read by a live process, or a
                # different venue. Adopting it would import a simulated drawdown
                # high-water mark, a simulated halt, or another venue's cooldowns
                # into live. Start from defaults, say so, and let the first good
                # cycle re-stamp the file.
                current = self.cfg.state_scope
                log.error(
                    "engine state at %s belongs to %s@%s but this process is %s@%s; "
                    "ignoring the stored risk state and halt flags",
                    path,
                    saved_scope.get("mode"), saved_scope.get("venue"),
                    current["mode"], current["venue"],
                )
                self.journal.event(
                    "state_scope_mismatch", saved=saved_scope, current=current
                )
                self.risk.restore({})
                self._last_decision_bar = None
                return
            risk_payload = payload.get("risk") or {}
            if not isinstance(risk_payload, dict):
                raise ValueError(f"risk state is a {type(risk_payload).__name__}, not an object")
            self.risk.restore(risk_payload)
            last_bar = payload.get("last_decision_bar")
            self._last_decision_bar = int(last_bar) if last_bar is not None else None
            saved_intents = payload.get("unknown_intents")
            if isinstance(saved_intents, dict):
                self._unknown_intents = {
                    str(pair): dict(intent)
                    for pair, intent in saved_intents.items()
                    if isinstance(intent, dict)
                }
                if self._unknown_intents:
                    log.error(
                        "restored %d unresolved order intent(s) (%s); they will be "
                        "re-queried this cycle and their stops are kept until they are",
                        len(self._unknown_intents),
                        ", ".join(sorted(self._unknown_intents)),
                    )
        except Exception as exc:
            # A well-formed-JSON file with a wrong *type* in it used to abort
            # bootstrap() with `_ready` still False, so `_persist` refused to
            # write and `Restart=always` turned one bad field into a restart loop
            # that never traded. Losing the risk state is bad; never starting is
            # worse. Start from the defaults, loudly, and let the file be
            # rewritten by the first successful cycle.
            log.error("could not read engine state (%s); starting from default risk state", exc)
            self.risk.restore({})
            self._last_decision_bar = None
            return
        if self.risk.halted:
            log.error("restored a HALTED state (%s); the kill switch requires a deliberate reset", self.risk.halt_reason)

    def _persist(self) -> None:
        import json

        if not self._ready:
            # Bootstrap never completed, so the in-memory book and risk state are
            # whatever the constructors left behind -- empty and default. Writing
            # them would destroy the real files on disk. `shutdown()` persists
            # from a `finally:`, so this guard is what makes a failed startup
            # non-destructive.
            log.error("not persisting state: bootstrap has not completed (a startup failure must not overwrite the stored book)")
            return

        self.book.save()
        try:
            payload = {
                "saved_ms": int(time.time() * 1000),
                # Which runtime wrote this. Without it, a mock-written file is
                # indistinguishable from a live one and gets loaded by both.
                "scope": self.cfg.state_scope,
                "risk": self.risk.snapshot(),
                # Orders the venue has not yet explained. Without this a restart
                # forgets that an order may exist, and the position it created is
                # later adopted with no stop.
                "unknown_intents": self._unknown_intents,
                "last_decision_bar": self._last_decision_bar,
                "stats": self.stats.to_dict(),
                "universe": list(self.universe),
            }
            path = self._state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            log.error("could not persist engine state: %s", exc)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def run(self, max_cycles: Optional[int] = None) -> EngineStats:
        """Run until stopped. One failing cycle never kills the process."""
        self.bootstrap()
        if self._seed_paths:
            self.seed_history()

        while not self._shutting_down:
            started = time.time()
            try:
                self.step()
                self.stats.consecutive_failures = 0
            except KeyboardInterrupt:
                log.info("interrupted")
                break
            except Exception as exc:
                self.stats.consecutive_failures += 1
                log.exception("cycle failed (%d in a row)", self.stats.consecutive_failures)
                self.journal.error("cycle", str(exc), consecutive_failures=self.stats.consecutive_failures)
                if self.stats.consecutive_failures >= 10:
                    # Give up and let the supervisor restart with a fresh clock sync.
                    log.error("10 consecutive failures; exiting so the service manager can restart")
                    self.journal.error("cycle", "aborting after 10 consecutive failures")
                    break
                time.sleep(min(2 ** self.stats.consecutive_failures, 60))
                continue

            if max_cycles is not None and self.stats.cycles >= max_cycles:
                log.info("reached max_cycles=%d", max_cycles)
                break
            elapsed = time.time() - started
            time.sleep(max(0.0, self.cfg.loop_interval_sec - elapsed))

        return self.stats

    def shutdown(self, flatten: bool = False) -> None:
        """Stop cleanly, optionally closing every position."""
        now_ms = int(time.time() * 1000)
        try:
            if flatten:
                current = PortfolioView(nav=0.0, cash_usd=0.0, positions=self.book.held())
                self._flatten(now_ms, reason="shutdown")
            self.journal.event("shutdown", stats=self.stats.to_dict(), metrics=self.tracker.metrics().to_dict())
        finally:
            self._persist()
            self.journal.close()
