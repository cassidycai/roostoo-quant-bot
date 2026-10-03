"""Portfolio state, position sizing, exposure caps and protective exits.

**Ownership note.** Rules 1-3 are this author's scope. This module is the
*generic harness* the remaining rules plug into: it computes NAV, tracks
positions across restarts, sizes orders and refuses ones that breach a cap. The
defaults already encode the team's playbook numbers, so the bot is runnable and
safe today:

===============  ==========================================
Rule 7           ``risk_per_trade_pct`` -> 0.5% NAV risked per trade
Rule 8           ``max_pair_weight``    -> 15% NAV per coin
Rule 9           ``max_gross_exposure`` -> 60% NAV gross
Rule 10          ``max_open_positions`` -> 4 concurrent coins
Rule 11          ``max_daily_loss_pct`` -> -2% day halts new entries
Rule 12          ``cooldown_bars``      -> 2 bars after an exit
Rule 5           ``stop_atr_mult``      -> stop at 1.5 x ATR(14)
Rule 6           ``max_hold_bars``      -> time stop after 12 bars
===============  ==========================================

Sizing is risk-first, which is the point of Rule 7: the position is chosen so
that being stopped out loses ``risk_per_trade_pct`` of NAV, and only then capped
by Rules 8 and 9. Whenever the risk budget binds, the team's 0.5% loss limit is
exact rather than approximate, and the caps act as a backstop for low-volatility
coins where an ATR stop would otherwise imply an enormous position.

Replacing any of this means editing one class; nothing in the engine assumes the
current policy.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .candles import bar_index
from .config import Config
from .metrics import MetricTracker
from .models import Position, Ticker
from .strategies.base import ENTER_LONG, ENTER_SHORT, EXIT_LONG, EXIT_SHORT, Signal

log = logging.getLogger(__name__)


def _as_finite_float(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion that never raises and never returns NaN/inf.

    Used on every field read back from a state file. A wrong type in one field of
    ``positions.json`` or ``engine_state.json`` must not be able to stop the bot
    from starting at all.
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        log.error("ignoring non-numeric state value %r; using %r", value, default)
        return default
    if not math.isfinite(result):
        log.error("ignoring non-finite state value %r; using %r", value, default)
        return default
    return result


def _as_optional_finite(value: Any) -> Optional[float]:
    """Like :func:`_as_finite_float`, but unset/blank stays ``None``.

    A stop price is a level or nothing. A string or a NaN here used to be stored
    verbatim and then compared against a float on every loop, which raised out of
    ``protective_exits`` -- ten cycles of that and the runner aborts the process.
    """
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        log.error("ignoring non-numeric price level %r", value)
        return None
    if not math.isfinite(result):
        log.error("ignoring non-finite price level %r", value)
        return None
    return result


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------


def portfolio_nav(cash_usd_total: float, positions: dict[str, Position]) -> float:
    """Net asset value, marked to market.

    ``cash_usd_total`` includes USD locked as short collateral, so short P&L must
    be added separately: a short's collateral stays on the books at face value
    while its unrealised P&L floats. Longs are valued at the mark, which already
    contains their P&L.
    """
    nav = float(cash_usd_total)
    for position in positions.values():
        if position.is_short:
            nav += position.unrealized_pnl
        else:
            nav += position.quantity * (position.mark_price or position.avg_price)
    return nav


@dataclass
class PortfolioView:
    """Immutable-ish snapshot handed to the risk layer."""

    nav: float
    cash_usd: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.notional) if p.notional else 0.0 for p in self.positions.values())

    @property
    def gross_pct(self) -> float:
        return self.gross_exposure / self.nav if self.nav > 0 else 0.0

    def weight(self, pair: str) -> float:
        position = self.positions.get(pair)
        if position is None or self.nav <= 0:
            return 0.0
        return abs(position.notional) / self.nav

    def open_pairs(self) -> set[str]:
        return {p for p, pos in self.positions.items() if pos.quantity > 0 or pos.is_short}

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": round(self.nav, 4),
            "cash_usd": round(self.cash_usd, 4),
            "gross_exposure": round(self.gross_exposure, 4),
            "gross_pct": round(self.gross_pct, 6),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "positions": {
                p: {
                    "qty": round(pos.quantity, 10),
                    "avg": round(pos.avg_price, 8),
                    "mark": round(pos.mark_price, 8),
                    "short": pos.is_short,
                    "weight": round(self.weight(p), 6),
                }
                for p, pos in self.positions.items()
            },
        }


# ---------------------------------------------------------------------------
# Position book (survives restarts)
# ---------------------------------------------------------------------------


class PositionBook:
    """Tracks cost basis and protective levels for every open position.

    The exchange API reports balances but not *when* or *at what price* a
    position was opened, and Rules 5/6/12 all need that history. The book is
    therefore persisted to disk: an unattended 14-day run will restart, and
    losing the entry price would silently disable every stop and cooldown.
    """

    def __init__(self, path: Optional[str | Path] = None, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg
        self.positions: dict[str, Position] = {}
        self.path = Path(path) if path else None
        #: Set only by a *successful* ``load()``. See ``save()`` for why the
        #: difference between "loaded and empty" and "never loaded" matters.
        self._load_attempted = False

    def lot_for(self, pair: str) -> float:
        """Smallest tradable increment, when a config with venue info is available."""
        if self.cfg is None:
            return 1e-9
        return self.cfg.lot_for(pair)

    # -- mutation --------------------------------------------------------
    def mark(self, tickers: dict[str, Ticker]) -> None:
        for pair, position in self.positions.items():
            ticker = tickers.get(pair)
            if ticker is not None and ticker.mid > 0:
                position.update_mark(ticker.mid)

    def apply_spot_buy(self, pair: str, quantity: float, price: float, ts_ms: int, stop_price: Optional[float] = None) -> Position:
        position = self.positions.get(pair)
        if position is None or position.is_short:
            position = Position(pair=pair, opened_ts_ms=ts_ms)
            self.positions[pair] = position
        total = position.quantity + quantity
        if total > 0:
            position.avg_price = (position.avg_price * position.quantity + price * quantity) / total
        position.quantity = total
        position.mark_price = price
        position.peak_price = max(position.peak_price, price)
        position.stop_price = stop_price if stop_price is not None else position.stop_price
        return position

    def apply_spot_sell(self, pair: str, quantity: float, price: float) -> Optional[Position]:
        position = self.positions.get(pair)
        if position is None:
            return None
        position.quantity = max(0.0, position.quantity - quantity)
        position.mark_price = price
        if position.quantity <= 1e-12:
            self.positions.pop(pair, None)
            return None
        return position

    def apply_short_open(
        self, pair: str, quantity: float, entry_price: float, collateral: float, ts_ms: int, stop_price: Optional[float] = None
    ) -> Position:
        position = self.positions.get(pair)
        if position is None or not position.is_short:
            position = Position(pair=pair, is_short=True, opened_ts_ms=ts_ms)
            self.positions[pair] = position
        total = position.quantity + quantity
        if total > 0:
            position.avg_price = (position.avg_price * position.quantity + entry_price * quantity) / total
        position.quantity = total
        position.collateral += collateral
        position.mark_price = entry_price
        position.peak_price = entry_price
        position.stop_price = stop_price if stop_price is not None else position.stop_price
        return position

    def apply_short_close(self, pair: str, quantity: float, price: float) -> Optional[Position]:
        position = self.positions.get(pair)
        if position is None:
            return None
        if position.quantity > 0:
            fraction = min(1.0, quantity / position.quantity)
            position.collateral *= 1.0 - fraction
        position.quantity = max(0.0, position.quantity - quantity)
        position.mark_price = price
        if position.quantity <= 1e-12:
            self.positions.pop(pair, None)
            return None
        return position

    def get(self, pair: str) -> Optional[Position]:
        return self.positions.get(pair)

    def held(self) -> dict[str, Position]:
        return {p: pos for p, pos in self.positions.items() if pos.quantity > 0 or pos.is_short}

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {
                pair: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "mark_price": pos.mark_price,
                    "is_short": pos.is_short,
                    "collateral": pos.collateral,
                    "stop_price": pos.stop_price,
                    "take_profit_price": pos.take_profit_price,
                    "peak_price": pos.peak_price,
                    "opened_ts_ms": pos.opened_ts_ms,
                }
                for pair, pos in self.positions.items()
            }
        }

    def _existing_book_is_non_empty(self) -> bool:
        """Is there a real book on disk that a default-constructed book would destroy?"""
        try:
            if not self.path or not self.path.is_file():
                return False
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return bool((payload or {}).get("positions"))
        except Exception:
            # Unreadable or corrupt: treat it as non-empty so we never clobber it.
            return True

    def _scope_mismatch(self, payload: Any) -> Optional[tuple[str, str]]:
        """``(written_by, this_process)`` when the file belongs to another runtime.

        A file with no stamp is accepted: state written before this check existed
        has to keep working, and the first run after the upgrade re-stamps it.
        Only an explicit contradiction is refused, because that is the case where
        adopting the contents would be wrong rather than merely incomplete.
        """
        if self.cfg is None or not isinstance(payload, dict):
            return None
        saved = payload.get("scope")
        if not isinstance(saved, dict):
            return None
        current = self.cfg.state_scope
        if all(saved.get(key) == current[key] for key in current):
            return None
        return (
            f"{saved.get('mode', '?')}@{saved.get('venue', '?')}",
            f"{current['mode']}@{current['venue']}",
        )

    def save(self) -> None:
        if not self.path:
            return
        # Never let an empty in-memory book overwrite a real file on disk. The
        # engine loads state at the very start of bootstrap(); if that never ran
        # -- a network blip, an exception before load, a partial startup -- then
        # this object is still empty *by default*, and writing it would silently
        # erase every stop level, cost basis, cooldown and the drawdown
        # high-water mark. A book whose positions were genuinely closed still
        # writes an empty file, because that path goes through load() first.
        if not self._load_attempted and self._existing_book_is_non_empty():
            log.error(
                "refusing to overwrite the non-empty position book at %s: it was never "
                "loaded in this process, so the in-memory book is empty by default",
                self.path,
            )
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = self.to_dict()
            # Stamp which runtime wrote this. mock and live share the default
            # journal directory, so without the stamp a simulated book is
            # indistinguishable from a real one. See Config.state_scope.
            if self.cfg is not None:
                payload["scope"] = self.cfg.state_scope
            tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            tmp.replace(self.path)  # atomic: never leave a half-written state file
        except Exception as exc:
            log.error("could not persist position book: %s", exc)

    def load(self) -> bool:
        if not self.path or not self.path.is_file():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("could not read position book (%s); starting flat", exc)
            return False
        mismatch = self._scope_mismatch(payload)
        if mismatch is not None:
            saved, current = mismatch
            log.error(
                "refusing to load the position book at %s: it was written by %s, this "
                "process is %s. Starting flat so one venue's positions and stops are not "
                "adopted by another. Point JOURNAL_DIR elsewhere, or delete the file, if "
                "switching venues is what you intended.",
                self.path, saved, current,
            )
            self.positions = {}
            self._load_attempted = True  # we did look; a flat book may now replace it
            return False
        self.positions = {}
        for pair, row in (payload.get("positions") or {}).items():
            if not isinstance(row, dict):
                log.error("ignoring malformed position row for %s (%r)", pair, row)
                continue
            try:
                quantity = _as_finite_float(row.get("quantity"), 0.0)
                collateral = _as_finite_float(row.get("collateral"), 0.0)
                # A row holding no quantity and no collateral is a leftover, not a
                # position: `{"quantity": null}` used to become a zero-quantity
                # entry that reads as an open slot. Half a lot is the same dust
                # threshold the balance reconciliation uses.
                lot = self.lot_for(pair)
                if quantity <= 0.5 * lot and collateral <= 0.0:
                    log.error("dropping dust/empty position row for %s (quantity=%r)", pair, row.get("quantity"))
                    continue
                self.positions[pair] = Position(
                    pair=pair,
                    quantity=quantity,
                    avg_price=_as_finite_float(row.get("avg_price"), 0.0),
                    mark_price=_as_finite_float(row.get("mark_price"), 0.0),
                    is_short=bool(row.get("is_short", False)),
                    collateral=collateral,
                    stop_price=_as_optional_finite(row.get("stop_price")),
                    take_profit_price=_as_optional_finite(row.get("take_profit_price")),
                    peak_price=_as_finite_float(row.get("peak_price"), 0.0),
                    opened_ts_ms=int(_as_finite_float(row.get("opened_ts_ms"), 0.0)),
                )
            except Exception as exc:
                # One unreadable row must not cost us the whole book -- the other
                # rows hold stops and cost bases the API cannot report back.
                log.error("could not restore position %s (%s); skipping that row", pair, exc)
                continue
        log.info("restored %d position(s) from %s", len(self.positions), self.path)
        self._load_attempted = True
        return True


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass
class SizingResult:
    notional: float
    quantity: float
    stop_price: Optional[float]
    risk_amount: float
    binding: str  # which constraint decided the size

    def to_dict(self) -> dict[str, Any]:
        return {
            "notional": round(self.notional, 4),
            "quantity": round(self.quantity, 10),
            "stop_price": None if self.stop_price is None else round(self.stop_price, 8),
            "risk_amount": round(self.risk_amount, 4),
            "binding": self.binding,
        }


class PositionSizer:
    """Risk-first sizing with the team's caps as a backstop."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def stop_distance(self, price: float, atr: Optional[float]) -> Optional[float]:
        """Absolute stop distance, ATR-based when available (Rule 5).

        A non-finite ATR counts as *unavailable*. ``atr > 0`` is False for NaN, so
        the old test fell through to the no-stop branch and the caller then had no
        way to tell "the stop is off by configuration" from "the ATR was junk" --
        it approved a full-size entry with no stop at all.
        """
        if self.cfg.stop_atr_mult and atr is not None:
            try:
                atr_value = float(atr)
            except (TypeError, ValueError):
                atr_value = 0.0
            if math.isfinite(atr_value) and atr_value > 0:
                return float(self.cfg.stop_atr_mult) * atr_value
        if self.cfg.stop_loss_pct:
            return price * float(self.cfg.stop_loss_pct)
        return None

    def slot_notional(self, nav: float) -> float:
        """Largest notional one position may take on cap grounds alone."""
        pair_cap = nav * self.cfg.max_pair_weight
        gross_share = nav * self.cfg.max_gross_exposure / max(self.cfg.max_open_positions, 1)
        return min(pair_cap, gross_share)

    def size(
        self,
        *,
        nav: float,
        price: float,
        atr: Optional[float] = None,
        gross_budget_left: Optional[float] = None,
        is_short: bool = False,
    ) -> SizingResult:
        if not (math.isfinite(price) and math.isfinite(nav)) or price <= 0 or nav <= 0:
            return SizingResult(0.0, 0.0, None, 0.0, "invalid_input")

        slot = self.slot_notional(nav)
        cap = slot if gross_budget_left is None else min(slot, max(0.0, gross_budget_left))
        binding = "pair_cap" if cap == slot else "gross_cap"

        distance = self.stop_distance(price, atr)
        risk_amount = nav * self.cfg.risk_per_trade_pct
        if distance and distance > 0:
            risk_notional = risk_amount / distance * price
            if risk_notional < cap:
                cap = risk_notional
                binding = "risk_budget"
        else:
            # No stop distance, so Rule 7 cannot be expressed and the position
            # would be sized on the caps alone -- strictly *larger* than the risk
            # budget allows. That is the opposite of what "risk-first" means, so
            # a stop that was *wanted* but could not be computed (a non-finite or
            # missing ATR) refuses the entry. An operator who deliberately turned
            # every stop off with STOP_ATR_MULT=none still gets the old
            # caps-only fallback, reported honestly as "+no_stop".
            stop_was_requested = bool(self.cfg.stop_atr_mult) or bool(self.cfg.stop_loss_pct)
            if stop_was_requested and self.cfg.refuse_entry_without_stop:
                return SizingResult(0.0, 0.0, None, 0.0, "no_stop_available")
            risk_amount = 0.0
            binding += "+no_stop"

        if distance is not None and distance > 0:
            stop_price: Optional[float] = price + distance if is_short else price - distance
        else:
            stop_price = None
        if stop_price is not None and not math.isfinite(stop_price):
            return SizingResult(0.0, 0.0, None, 0.0, "invalid_stop")

        cap = max(cap, 0.0)
        quantity = cap / price
        if not (math.isfinite(quantity) and quantity > 0):
            return SizingResult(0.0, 0.0, None, 0.0, "invalid_quantity")

        return SizingResult(
            notional=cap,
            quantity=quantity,
            stop_price=stop_price,
            risk_amount=risk_amount if stop_price else 0.0,
            binding=binding,
        )


# ---------------------------------------------------------------------------
# Approved actions
# ---------------------------------------------------------------------------


@dataclass
class ApprovedAction:
    """A risk-approved intent, ready for the execution layer."""

    pair: str
    action: str
    quantity: float = 0.0
    collateral: float = 0.0
    notional: float = 0.0
    order_type: str = "MARKET"
    price: Optional[float] = None
    reason: str = ""
    stop_price: Optional[float] = None
    risk_amount: float = 0.0
    binding: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.action in (ENTER_LONG, ENTER_SHORT)

    @property
    def is_exit(self) -> bool:
        return self.action in (EXIT_LONG, EXIT_SHORT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "action": self.action,
            "quantity": round(self.quantity, 10),
            "collateral": round(self.collateral, 4),
            "notional": round(self.notional, 4),
            "order_type": self.order_type,
            "price": None if self.price is None else round(self.price, 8),
            "reason": self.reason,
            "stop_price": None if self.stop_price is None else round(self.stop_price, 8),
            "risk_amount": round(self.risk_amount, 4),
            "binding": self.binding,
            "meta": self.meta,
        }


@dataclass
class RiskDecision:
    approved: list[ApprovedAction] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""
    notes: list[str] = field(default_factory=list)
    nav: float = 0.0
    gross_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": [a.to_dict() for a in self.approved],
            "rejected": [{"pair": p, "reason": r} for p, r in self.rejected],
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "notes": self.notes,
            "nav": round(self.nav, 4),
            "gross_pct": round(self.gross_pct, 6),
        }


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------


class RiskManager:
    """Enforces portfolio rules and produces protective exits.

    Lifecycle per bar::

        risk.observe(nav, now_ms)                    # update day/peak, trip halts
        exits = risk.protective_exits(...)           # Rules 5 and 6
        decision = risk.evaluate(signals, ...)       # Rules 7-12
        risk.record_exit(pair, bar_idx)              # Rule 12 cooldown
    """

    def __init__(self, cfg: Config, sizer: Optional[PositionSizer] = None) -> None:
        self.cfg = cfg
        self.sizer = sizer or PositionSizer(cfg)
        self.peak_nav = 0.0
        self.day_id: Optional[int] = None
        self.day_start_nav = 0.0
        self.daily_halt = False
        self.daily_halt_reason = ""
        self.halted = False
        self.halt_reason = ""
        self.last_exit_bar: dict[str, int] = {}
        self.entries_today = 0
        self.last_nav = 0.0

    # -- observation -----------------------------------------------------
    def observe(self, nav: float, now_ms: int) -> None:
        """Update the day boundary and high-water mark; trip halts if breached.

        A non-finite NAV is *ignored* rather than folded into the reference
        values. ``nan`` compares False against everything, so the old code turned
        ``peak_nav`` into NaN on the first bad cycle and every later
        ``if self.peak_nav > 0`` guard stayed False for the rest of the run --
        silently disabling the permanent kill switch. A NaN also survived into
        the persisted state, where ``json.dumps`` emits a bare ``NaN`` token that
        strict JSON parsers reject.

        A NAV at or below zero is treated as a halt, not as a drawdown to be
        measured: the account is gone, and the guards below would otherwise
        suppress both halts because the reference is no longer positive.
        """
        from .candles import trading_day_id

        if not math.isfinite(nav):
            log.error("non-finite NAV (%r); ignoring it for the risk reference values", nav)
            return

        if nav <= 0:
            if not self.halted:
                self.halted = True
                self.halt_reason = f"NAV {nav!r} is non-positive"
                log.error("KILL SWITCH: %s", self.halt_reason)
            self.last_nav = nav
            return

        if self.peak_nav <= 0:
            self.peak_nav = nav
        self.peak_nav = max(self.peak_nav, nav)

        day = trading_day_id(now_ms, self.cfg.trading_day_offset_hours)
        if self.day_id != day:
            self.day_id = day
            self.day_start_nav = nav
            if self.daily_halt:
                log.info("new trading day %s: daily halt cleared", day)
            self.daily_halt = False
            self.daily_halt_reason = ""
            self.entries_today = 0

        # Rule 11: a -2% day stops new risk until the next UTC+offset day.
        if self.day_start_nav > 0:
            day_return = nav / self.day_start_nav - 1.0
            if day_return <= -self.cfg.max_daily_loss_pct and not self.daily_halt:
                self.daily_halt = True
                self.daily_halt_reason = (
                    f"daily loss {day_return * 100:.2f}% <= -{self.cfg.max_daily_loss_pct * 100:.2f}%"
                )
                log.warning("Rule 11 halt: %s", self.daily_halt_reason)

        self.last_nav = nav

        # Portfolio kill switch. Deliberately separate from Rule 11: this one is
        # permanent for the run and flattens the book.
        if self.peak_nav > 0:
            drawdown = 1.0 - nav / self.peak_nav
            if drawdown >= self.cfg.max_drawdown_pct and not self.halted:
                self.halted = True
                self.halt_reason = (
                    f"drawdown {drawdown * 100:.2f}% >= {self.cfg.max_drawdown_pct * 100:.2f}% of peak NAV"
                )
                log.error("KILL SWITCH: %s", self.halt_reason)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak NAV, as a positive fraction."""
        if self.peak_nav <= 0 or self.last_nav <= 0:
            return 0.0
        return max(0.0, 1.0 - self.last_nav / self.peak_nav)

    def cooldown_blocked(self, bar_idx: int) -> set[str]:
        """Rule 12: pairs closed within the last ``cooldown_bars`` bars.

        The comparison is inclusive of the bar that is ``cooldown_bars`` old. With
        ``<`` an exit recorded on bar *b* stopped blocking at *b + N*, so only
        ``N - 1`` bars were actually skipped and the documented "no re-entry for 2
        bars" was a 1-bar wait in practice.
        """
        if self.cfg.cooldown_bars <= 0:
            return set()
        return {
            pair
            for pair, exit_bar in self.last_exit_bar.items()
            if bar_idx - exit_bar <= self.cfg.cooldown_bars
        }

    def record_exit(self, pair: str, bar_idx: int) -> None:
        self.last_exit_bar[pair] = bar_idx

    def blocked_pairs(self, bar_idx: int) -> set[str]:
        return self.cooldown_blocked(bar_idx)

    # -- protective exits (Rules 5 and 6) --------------------------------
    def protective_exits(
        self,
        positions: dict[str, Position],
        tickers: dict[str, Ticker],
        now_ms: int,
    ) -> list[Signal]:
        """Stops and time stops. These are checked every loop, not per bar.

        A stop is worthless if it is only evaluated on the half hour: Rule 5 says
        exit when the adverse move exceeds 1.5 x ATR, and the price can do that
        between two bar closes. The time stop (Rule 6) is bar-counted.
        """
        out: list[Signal] = []
        now_bar = bar_index(now_ms, self.cfg.bar_seconds)
        for pair, position in positions.items():
            if not (position.quantity > 0 or position.is_short):
                continue
            ticker = tickers.get(pair)
            mark = ticker.mid if ticker and ticker.mid > 0 else position.mark_price

            # Rule 6 is checked BEFORE the mark is validated. It is the only exit
            # that does not need a price, so it must not be gated behind one: a
            # position whose mark went to zero (an adopted holding, a phantom
            # fill, a venue that stopped quoting the pair) used to skip every
            # exit here *and* every exit in the execution layer, which refused to
            # sell without a usable price. The position became permanently stuck,
            # including for the kill switch.
            if self.cfg.max_hold_bars and position.opened_ts_ms:
                held = now_bar - bar_index(position.opened_ts_ms, self.cfg.bar_seconds)
                if held >= int(self.cfg.max_hold_bars):
                    out.append(
                        Signal(
                            pair,
                            EXIT_SHORT if position.is_short else EXIT_LONG,
                            reason=f"time stop: held {held} bars >= {self.cfg.max_hold_bars}",
                            meta={"trigger": "time_stop", "bars_held": held},
                        )
                    )
                    continue

            if not (mark > 0) or not math.isfinite(mark):
                # No usable mark and the time stop has not fired yet. Nothing can
                # be evaluated from here, but say so rather than silence.
                log.warning(
                    "%s: no usable mark (mark_price=%r); cannot evaluate the stop this loop",
                    pair,
                    position.mark_price,
                )
                continue

            # Rule 5: ATR stop.
            if position.stop_price:
                breached = mark >= position.stop_price if position.is_short else mark <= position.stop_price
                if breached:
                    basis = " on collateral" if position.is_short else ""
                    out.append(
                        Signal(
                            pair,
                            EXIT_SHORT if position.is_short else EXIT_LONG,
                            reason=(
                                f"stop hit: mark {mark:.6f} vs stop {position.stop_price:.6f} "
                                f"({position.unrealized_pct * 100:+.2f}%{basis})"
                            ),
                            meta={"trigger": "stop", "mark": mark, "stop": position.stop_price},
                        )
                    )
                    continue

            # Optional trailing stop (off by default; the playbook has none).
            if self.cfg.trailing_stop_pct and not position.is_short and position.peak_price > 0:
                drop = 1.0 - mark / position.peak_price
                if drop >= self.cfg.trailing_stop_pct:
                    out.append(
                        Signal(
                            pair,
                            EXIT_LONG,
                            reason=f"trailing stop: {drop * 100:.2f}% off peak {position.peak_price:.6f}",
                            meta={"trigger": "trailing_stop"},
                        )
                    )
                    continue

            # Optional fixed take-profit (off by default).
            if self.cfg.take_profit_pct and position.avg_price > 0:
                move = (mark / position.avg_price - 1.0) * (-1 if position.is_short else 1)
                if move >= self.cfg.take_profit_pct:
                    out.append(
                        Signal(
                            pair,
                            EXIT_SHORT if position.is_short else EXIT_LONG,
                            reason=f"take profit {move * 100:.2f}% >= {self.cfg.take_profit_pct * 100:.2f}%",
                            meta={"trigger": "take_profit"},
                        )
                    )
                    continue

        return out

    # -- approval --------------------------------------------------------
    def evaluate(
        self,
        signals: Iterable[Signal],
        *,
        view: PortfolioView,
        tickers: dict[str, Ticker],
        now_ms: int,
        bar_idx: int,
        committed_pairs: Optional[Iterable[str]] = None,
        committed_notional: float = 0.0,
    ) -> RiskDecision:
        """Approve or reject this bar's intents.

        ``committed_pairs``/``committed_notional`` describe orders that are
        already in flight but have not filled yet. They must be counted as if
        they were positions. Without this, two consecutive bars can each approve
        a full-size entry for the same pair -- the account is still flat when the
        second decision is made -- and the resulting position is double the cap.
        That is a sizing bug, not a rounding error, so it is handled here where
        every caller inherits the protection.
        """
        decision = RiskDecision(nav=view.nav, gross_pct=view.gross_pct)
        if self.halted:
            decision.halted = True
            decision.halt_reason = self.halt_reason

        in_flight = {p for p in (committed_pairs or ())}
        committed = max(0.0, float(committed_notional))
        cooldown = self.cooldown_blocked(bar_idx)
        gross_cap = view.nav * self.cfg.max_gross_exposure
        # Track projected exposure as approvals accumulate, so a bar that fires
        # four entries cannot collectively overshoot Rule 9. Rule 9 is tested
        # against this projected figure rather than a budget computed once up
        # front: an exit approved earlier in the same batch frees its capital,
        # and a stale budget kept rejecting entries that now fit (the slot was
        # freed by Rule 10 but the money was not freed by Rule 9).
        projected_gross = view.gross_exposure + committed
        projected_pairs = set(view.open_pairs()) | in_flight

        for signal in signals:
            pair = signal.pair
            if signal.is_exit:
                position = view.positions.get(pair)
                if position is None:
                    decision.rejected.append((pair, "exit for a position that is not open"))
                    continue
                decision.approved.append(
                    ApprovedAction(
                        pair=pair,
                        action=signal.action,
                        quantity=position.quantity,
                        collateral=position.collateral,
                        notional=abs(position.notional),
                        reason=signal.reason,
                        meta=dict(signal.meta),
                    )
                )
                projected_pairs.discard(pair)
                projected_gross -= abs(position.notional)
                continue

            # --- entries -------------------------------------------------
            if self.halted:
                decision.rejected.append((pair, f"kill switch active: {self.halt_reason}"))
                continue
            if self.daily_halt:
                decision.rejected.append((pair, f"Rule 11 daily halt: {self.daily_halt_reason}"))
                continue
            if pair in cooldown:
                bars_left = self.cfg.cooldown_bars - (bar_idx - self.last_exit_bar.get(pair, bar_idx))
                decision.rejected.append((pair, f"Rule 12 cooldown: {bars_left} bar(s) left"))
                continue
            if pair in projected_pairs:
                decision.rejected.append((pair, "already holding this pair"))
                continue
            if len(projected_pairs) >= self.cfg.max_open_positions:
                decision.rejected.append((pair, f"Rule 10: max {self.cfg.max_open_positions} positions"))
                continue
            if gross_cap - projected_gross <= 0:
                decision.rejected.append((pair, f"Rule 9: gross exposure at {self.cfg.max_gross_exposure:.0%} NAV"))
                continue

            ticker = tickers.get(pair)
            price = ticker.mid if ticker and ticker.mid > 0 else 0.0
            if price <= 0:
                decision.rejected.append((pair, "no usable price"))
                continue

            atr = signal.meta.get("atr") if signal.meta else None
            is_short = signal.action == ENTER_SHORT
            sizing = self.sizer.size(
                nav=view.nav,
                price=price,
                atr=atr if isinstance(atr, (int, float)) else None,
                gross_budget_left=gross_cap - projected_gross,
                is_short=is_short,
            )
            if sizing.notional < self.cfg.min_order_notional:
                decision.rejected.append(
                    (pair, f"sized {sizing.notional:,.2f} < min order {self.cfg.min_order_notional:,.2f}")
                )
                continue

            decision.approved.append(
                ApprovedAction(
                    pair=pair,
                    action=signal.action,
                    quantity=sizing.quantity,
                    collateral=sizing.notional if is_short else 0.0,
                    notional=sizing.notional,
                    reason=signal.reason,
                    stop_price=sizing.stop_price,
                    risk_amount=sizing.risk_amount,
                    binding=sizing.binding,
                    meta={**dict(signal.meta), "strength": round(signal.strength, 4)},
                )
            )
            projected_pairs.add(pair)
            projected_gross += sizing.notional

        if self.halted:
            decision.notes.append("kill switch active: flattening")
        if self.daily_halt:
            decision.notes.append("Rule 11: no new entries today")
        return decision

    # -- emergency -------------------------------------------------------
    def flatten_all(self, positions: dict[str, Position]) -> list[ApprovedAction]:
        """Close everything. Used by the kill switch and at shutdown."""
        out: list[ApprovedAction] = []
        for pair, position in positions.items():
            if not (position.quantity > 0 or position.is_short):
                continue
            out.append(
                ApprovedAction(
                    pair=pair,
                    action=EXIT_SHORT if position.is_short else EXIT_LONG,
                    quantity=position.quantity,
                    collateral=position.collateral,
                    notional=abs(position.notional),
                    reason="flatten (kill switch or shutdown)",
                )
            )
        return out

    # -- persistence -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "peak_nav": self.peak_nav,
            "day_id": self.day_id,
            "day_start_nav": self.day_start_nav,
            "daily_halt": self.daily_halt,
            "daily_halt_reason": self.daily_halt_reason,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "last_exit_bar": {k: int(v) for k, v in self.last_exit_bar.items()},
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore risk state, tolerating a malformed field.

        Each field is coerced independently so one wrong type -- a string where a
        number belongs, a null quantity -- degrades that single value instead of
        aborting the whole start-up. An exception here used to escape
        ``TradingEngine._load_risk_state`` and stop ``bootstrap()`` before
        ``_ready`` was set, which under ``Restart=always`` is a restart loop that
        never trades.
        """
        peak = _as_finite_float(data.get("peak_nav"), 0.0)
        day_start = _as_finite_float(data.get("day_start_nav"), 0.0)
        self.peak_nav = peak
        self.day_start_nav = day_start
        day_id = data.get("day_id")
        try:
            self.day_id = int(day_id) if day_id is not None else None
        except (TypeError, ValueError):
            log.error("ignoring malformed day_id %r in the restored risk state", day_id)
            self.day_id = None
        self.daily_halt = bool(data.get("daily_halt", False))
        self.daily_halt_reason = str(data.get("daily_halt_reason", "") or "")
        self.halted = bool(data.get("halted", False))
        self.halt_reason = str(data.get("halt_reason", "") or "")
        last_exit: dict[str, int] = {}
        raw_exits = data.get("last_exit_bar")
        if isinstance(raw_exits, dict):
            for pair, bar in raw_exits.items():
                try:
                    last_exit[str(pair)] = int(bar)
                except (TypeError, ValueError):
                    log.error("ignoring malformed cooldown entry %s=%r", pair, bar)
        self.last_exit_bar = last_exit

    def attach_tracker(self, tracker: MetricTracker) -> None:
        """Keep the high-water mark consistent with the equity tracker."""
        self.peak_nav = max(self.peak_nav, tracker.peak_equity)
