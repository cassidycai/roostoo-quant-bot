"""Domain objects shared by the client, the engine and the backtester.

Everything is a plain dataclass with no third-party dependency, so the same
objects can be produced by the live REST client and by the offline simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(value: float, precision: int) -> str:
    """Format a number to a fixed precision without float repr noise.

    ``str(0.1 + 0.2)`` is ``0.30000000000000004``; the exchange rejects that
    against ``AmountPrecision`` and, worse, the signed payload would not match
    the value the server expects. Decimal round-down is always safe: it can
    only ever make an order slightly smaller, never larger.
    """
    if precision < 0:
        raise ValueError("precision must be >= 0")
    quant = Decimal(1).scaleb(-precision)
    dec = Decimal(str(float(value))).quantize(quant, rounding=ROUND_DOWN)
    return f"{dec:.{precision}f}"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradePair:
    """A tradable symbol as described by ``/v3/exchangeInfo``."""

    pair: str
    coin: str
    unit: str
    can_trade: bool
    price_precision: int
    amount_precision: int
    min_order: float

    @property
    def base(self) -> str:
        return self.coin

    @property
    def quote(self) -> str:
        return self.unit

    @classmethod
    def from_api(cls, pair: str, raw: dict[str, Any]) -> "TradePair":
        return cls(
            pair=pair,
            coin=str(raw.get("Coin", pair.split("/")[0])),
            unit=str(raw.get("Unit", "USD")),
            can_trade=bool(raw.get("CanTrade", False)),
            price_precision=int(raw.get("PricePrecision", 2)),
            amount_precision=int(raw.get("AmountPrecision", 6)),
            min_order=float(raw.get("MiniOrder", 0.0) or 0.0),
        )

    def round_qty(self, qty: float) -> str:
        return fmt(qty, self.amount_precision)

    def round_price(self, price: float) -> str:
        return fmt(price, self.price_precision)

    def min_qty_for(self, price: float) -> float:
        """Smallest quantity that satisfies the pair's ``MiniOrder`` notional."""
        if price <= 0:
            return math.inf
        return self.min_order / price


@dataclass(frozen=True)
class Ticker:
    """One row of ``/v3/ticker``."""

    pair: str
    last: float
    max_bid: float
    min_ask: float
    change_24h: float
    coin_volume: float
    unit_volume: float
    server_time_ms: int

    @property
    def mid(self) -> float:
        if self.max_bid > 0 and self.min_ask > 0:
            return (self.max_bid + self.min_ask) / 2.0
        return self.last

    @property
    def spread_bps(self) -> float:
        """Quoted spread in basis points -- a real cost we must not ignore."""
        mid = self.mid
        if mid <= 0 or self.min_ask <= 0 or self.max_bid <= 0:
            return 0.0
        return (self.min_ask - self.max_bid) / mid * 10_000.0

    @classmethod
    def from_api(cls, pair: str, raw: dict[str, Any], server_time_ms: int) -> "Ticker":
        return cls(
            pair=pair,
            last=float(raw.get("LastPrice", 0.0) or 0.0),
            max_bid=float(raw.get("MaxBid", 0.0) or 0.0),
            min_ask=float(raw.get("MinAsk", 0.0) or 0.0),
            change_24h=float(raw.get("Change", 0.0) or 0.0),
            coin_volume=float(raw.get("CoinTradeValue", 0.0) or 0.0),
            unit_volume=float(raw.get("UnitTradeValue", 0.0) or 0.0),
            server_time_ms=int(server_time_ms),
        )


@dataclass
class Bar:
    """A sampled price observation.

    The public API exposes no OHLCV history, so the bot builds its own bars by
    sampling the ticker on every loop. ``source`` records which field was used.
    """

    ts_ms: int
    price: float
    bid: float = 0.0
    ask: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {"ts_ms": self.ts_ms, "price": self.price, "bid": self.bid, "ask": self.ask}


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------


@dataclass
class WalletBalance:
    asset: str
    free: float
    locked: float

    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class Position:
    """Internal view of a spot holding (long) or a v6 short."""

    pair: str
    quantity: float = 0.0
    avg_price: float = 0.0
    mark_price: float = 0.0
    is_short: bool = False
    collateral: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    peak_price: float = 0.0
    opened_ts_ms: int = 0
    #: Venue order ids already applied to this position.
    #:
    #: The same fill can be presented twice -- the history query that resolves an
    #: UNKNOWN order may find a fill that was already booked normally, and a restart
    #: can replay it -- so applying is made idempotent by id rather than by hoping it
    #: only happens once. A tuple keeps the dataclass free of mutable defaults.
    applied_order_ids: tuple[str, ...] = ()

    @property
    def notional(self) -> float:
        return abs(self.quantity) * (self.mark_price or self.avg_price)

    @property
    def unrealized_pnl(self) -> float:
        if self.is_short:
            return self.quantity * (self.avg_price - self.mark_price)
        return self.quantity * (self.mark_price - self.avg_price)

    @property
    def unrealized_pct(self) -> float:
        """Return on the capital actually committed to this position."""
        basis = self.collateral if self.is_short else self.quantity * self.avg_price
        if basis <= 0:
            return 0.0
        return self.unrealized_pnl / basis

    def update_mark(self, price: float) -> None:
        self.mark_price = price
        if price > self.peak_price:
            self.peak_price = price


@dataclass
class OrderRequest:
    """An intent to trade, before risk approval or wire encoding."""

    pair: str
    side: str  # BUY | SELL
    quantity: float
    order_type: str = "MARKET"  # MARKET | LIMIT
    price: Optional[float] = None
    reason: str = ""
    target_weight: Optional[float] = None
    reduce_only: bool = False

    @property
    def notional(self) -> float:
        ref = self.price if self.price else 0.0
        return self.quantity * ref


@dataclass
class OrderResult:
    """Normalised ``/v3/place_order`` result.

    ``status == "UNKNOWN"`` means transport failed after the request was built:
    the order may exist. The engine reconciles instead of retrying.
    """

    pair: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: str  # FILLED | PENDING | CANCELED | REJECTED | UNKNOWN
    order_id: Optional[int] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    commission_coin: str = ""
    role: str = ""
    err_msg: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.status in ("FILLED", "PENDING")

    @classmethod
    def from_api(cls, pair: str, side: str, order_type: str, qty: float, raw: dict[str, Any]) -> "OrderResult":
        """Normalise an API payload.

        An absent ``Status`` must NOT be read as ``FILLED``. That default is what
        let a history row with no status -- and ``FilledQuantity=0`` -- reach the
        execution path as a completed fill, which then booked a full-size
        position at price zero. Unstated means unknown, and the execution layer
        refuses to act on an unknown or unfilled order.
        """
        detail = raw.get("OrderDetail") or {}
        status = str(detail.get("Status", "") or "").strip().upper() or "UNKNOWN"
        return cls(
            pair=str(detail.get("Pair", pair)),
            side=str(detail.get("Side", side)),
            order_type=str(detail.get("Type", order_type)),
            quantity=float(detail.get("Quantity", qty) or qty),
            price=float(detail.get("Price", 0.0) or 0.0),
            status=status,
            order_id=int(detail["OrderID"]) if detail.get("OrderID") is not None else None,
            filled_quantity=float(detail.get("FilledQuantity", 0.0) or 0.0),
            avg_fill_price=float(detail.get("FilledAverPrice", 0.0) or 0.0),
            commission=float(detail.get("CommissionChargeValue", 0.0) or 0.0),
            commission_coin=str(detail.get("CommissionCoin", "") or ""),
            role=str(detail.get("Role", "") or ""),
            raw=raw,
        )


@dataclass
class ShortPosition:
    """A ``/v6/short_positions`` row."""

    position_id: int
    pair: str
    entry_price: float
    quantity: float
    collateral: float
    current_price: float
    unrealized_pnl: float
    unrealized_pct: float
    position_value: float
    created_ts_ms: int

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ShortPosition":
        return cls(
            position_id=int(raw.get("ID", 0) or 0),
            pair=str(raw.get("Pair", "")),
            entry_price=float(raw.get("EntryPrice", 0.0) or 0.0),
            quantity=float(raw.get("ShortQty", 0.0) or 0.0),
            collateral=float(raw.get("Collateral", 0.0) or 0.0),
            current_price=float(raw.get("CurrentPrice", 0.0) or 0.0),
            unrealized_pnl=float(raw.get("UnrealizedPNL", 0.0) or 0.0),
            unrealized_pct=float(raw.get("UnrealizedPNLPct", 0.0) or 0.0),
            position_value=float(raw.get("PositionValue", 0.0) or 0.0),
            created_ts_ms=int(raw.get("CreateTimestamp", 0) or 0),
        )


@dataclass
class Fill:
    """A trade that actually happened -- the record we persist and are judged on."""

    ts_ms: int
    pair: str
    side: str
    quantity: float
    price: float
    fee: float
    order_id: Optional[int]
    role: str
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    def to_row(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "pair": self.pair,
            "side": self.side,
            "quantity": f"{self.quantity:.10f}",
            "price": f"{self.price:.10f}",
            "notional": f"{self.notional:.6f}",
            "fee": f"{self.fee:.6f}",
            "order_id": self.order_id if self.order_id is not None else "",
            "role": self.role,
            "reason": self.reason,
        }


@dataclass
class ExchangeInfo:
    is_running: bool
    initial_wallet: dict[str, float]
    pairs: dict[str, TradePair]

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ExchangeInfo":
        pairs_raw = raw.get("TradePairs") or {}
        return cls(
            is_running=bool(raw.get("IsRunning", False)),
            initial_wallet={
                str(k): float(v) for k, v in (raw.get("InitialWallet") or {}).items()
            },
            pairs={p: TradePair.from_api(p, v) for p, v in pairs_raw.items()},
        )
