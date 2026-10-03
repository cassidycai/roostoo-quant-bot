"""Environment-driven configuration.

Precedence: real process environment > ``.env`` file > defaults in this file.
A tiny hand-rolled ``.env`` parser keeps the runtime dependency-free.

The risk defaults deliberately mirror the team's playbook so that the harness is
safe out of the box, even before Rules 4-12 are wired in:

* ``max_pair_weight 0.15``  -- Rule 8  (no single coin above 15% of NAV)
* ``max_gross_exposure 0.60`` -- Rule 9  (gross exposure <= 60% of NAV)
* ``max_open_positions 4``  -- Rule 10 (at most four concurrent coins)
* ``max_daily_loss_pct 0.02`` -- Rule 11 (halt new risk after a -2% day)
* ``cooldown_bars 2``       -- Rule 12 (no re-entry for 2 bars after an exit)
* ``risk_per_trade_pct``    -- Rule 7  (max loss per trade = 0.5% of portfolio)
* ``stop_atr_mult``         -- Rule 5  (stop at 1.5 x ATR(14))
* ``max_hold_bars``         -- Rule 6  (time stop after 12 bars)

Note that 0.15 x 4 == 0.60: the per-coin cap and the gross cap bind at exactly
the same point, which is what makes equal-weight sizing across four slots
satisfy Rules 8, 9 and 10 simultaneously.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from .errors import ConfigError

DEFAULT_BASE_URL = "https://mock-api.roostoo.com"
DEFAULT_STRATEGY = "roostoo.strategies.mean_reversion:MeanReversionStrategy"

# Competition rulebook: 0.1% taker (market), 0.05% maker (limit).
DEFAULT_TAKER_FEE = 0.001
DEFAULT_MAKER_FEE = 0.0005


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse a minimal ``.env`` body. Supports `#` comments and quoted values."""
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Strip a trailing inline comment only when unquoted.
            hash_idx = value.find(" #")
            if hash_idx != -1:
                value = value[:hash_idx].strip()
        if key:
            out[key] = value
    return out


def load_dotenv(path: os.PathLike[str] | str = ".env", override: bool = False) -> dict[str, str]:
    """Load ``path`` into ``os.environ``. Never raises if the file is absent."""
    p = Path(path)
    if not p.is_file():
        return {}
    values = parse_dotenv(p.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _get(env: dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _as_float(env: dict[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be numeric, got {raw!r}") from exc


def _as_int(env: dict[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


#: Tokens that explicitly *disable* an optional setting, as opposed to leaving it
#: unset (which keeps the default). ``STOP_ATR_MULT=none`` has to mean "no ATR
#: stop", so it cannot share the defaulting path: an earlier version returned the
#: default here, silently re-enabling the very stop the operator disabled.
_DISABLE_TOKENS = ("none", "null", "off", "disabled", "no", "false")


def _as_optional_float(env: dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    raw = _get(env, key)
    if raw is None:
        return default
    if raw.strip().lower() in _DISABLE_TOKENS:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be numeric, or 'none' to disable it, got {raw!r}") from exc


def _as_optional_int(env: dict[str, str], key: str, default: Optional[int] = None) -> Optional[int]:
    raw = _get(env, key)
    if raw is None:
        return default
    if raw.strip().lower() in _DISABLE_TOKENS:
        return None
    try:
        return int(float(raw))
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, or 'none' to disable it, got {raw!r}") from exc


def _as_list(env: dict[str, str], key: str) -> tuple[str, ...]:
    raw = _get(env, key)
    if raw is None:
        return ()
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return tuple(p for p in parts if p)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # credentials
    api_key: str = ""
    secret_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    mock: bool = False

    # universe (Rule 1)
    pairs: tuple[str, ...] = ()
    max_pairs: int = 8
    max_spread_bps: float = 10.0  # 0.1% bid-ask spread ceiling
    depth_provider: str = "none"  # none | ticker | binance | roostoo
    depth_path: str = ""  # Roostoo depth path, once the organisers confirm one
    depth_band_pct: float = 0.005  # +/- 0.5% of mid
    depth_target_notional: float = 2_000.0  # the "$X" of Rule 1
    depth_liquidity_share: float = 0.02  # ticker-proxy assumption, see universe.py

    # loop
    bar_seconds: int = 1800  # 30-minute bars: the unit of Rules 2-6
    loop_interval_sec: float = 60.0
    history_window: int = 500
    request_timeout_sec: float = 15.0
    max_retries: int = 3
    min_request_interval_sec: float = 0.25

    # capital & portfolio risk (Rules 7-12)
    initial_capital: float = 100_000.0
    max_gross_exposure: float = 0.60
    max_pair_weight: float = 0.15
    max_open_positions: int = 4
    min_order_notional: float = 25.0
    risk_per_trade_pct: float = 0.005
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.20
    cooldown_bars: int = 2
    trading_day_offset_hours: int = 8  # Rule 11 day boundary: UTC+8
    vol_target_annual: float = 0.25
    stop_atr_mult: Optional[float] = 1.5  # Rule 5; None disables
    max_hold_bars: Optional[int] = 12  # Rule 6; None disables
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    crash_filter_pct: float = 0.12
    #: Refuse an entry whose stop distance could not be computed (a missing or
    #: non-finite ATR). Without this, an uncomputable stop silently produced a
    #: *full-size* position with no stop: Rule 7's 0.5% risk budget was reported
    #: as unenforceable and the size fell back to the per-pair cap.
    refuse_entry_without_stop: bool = True
    #: Post entries as resting maker orders instead of crossing the spread. The
    #: mean-reversion entry is the natural case for a passive bid -- the signal IS
    #: "price just fell hard", so a bid at the quote is often filled by the same
    #: move, and it pays 0.05% instead of 0.1% plus slippage. Measured effect on
    #: the round trip is 0.30% -> 0.25%, worth about +0.047% per trade: real, and
    #: about a seventh of the loss, not a fix (see docs/FINDINGS.md 6b.1).
    #: Exits always cross -- a stop that rests is not a stop.
    limit_entries: bool = False
    #: How far below the mid the bid sits. 0 bids at the mid, which is where the
    #: signal was born; positive values are more patient and fill less often.
    limit_entry_offset_bps: float = 0.0
    #: Cancel after this many bars unfilled. The measured fill rate is 95-99%
    #: within one bar, so a longer window mostly buys exposure to a reversion that
    #: has already happened.
    limit_entry_timeout_bars: int = 1

    # costs
    taker_fee: float = DEFAULT_TAKER_FEE
    maker_fee: float = DEFAULT_MAKER_FEE
    slippage_bps: float = 5.0

    # strategy plug-in
    strategy: str = DEFAULT_STRATEGY
    strategy_params: dict[str, Any] = field(default_factory=dict)

    # logging
    log_level: str = "INFO"
    journal_dir: str = "journal"
    log_dir: str = "logs"

    # metrics convention
    periods_per_year: int = 365
    risk_free_rate: float = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None, require_keys: bool = True) -> "Config":
        env = dict(os.environ if env is None else env)
        raw_params = _get(env, "STRATEGY_PARAMS", "{}") or "{}"
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"STRATEGY_PARAMS must be valid JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise ConfigError("STRATEGY_PARAMS must be a JSON object")

        cfg = cls(
            api_key=_get(env, "ROOSTOO_API_KEY", "") or "",
            secret_key=_get(env, "ROOSTOO_SECRET_KEY", "") or "",
            base_url=(_get(env, "ROOSTOO_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
            mock=_as_bool(_get(env, "ROOSTOO_MOCK"), False),
            pairs=_as_list(env, "ROOSTOO_PAIRS"),
            max_pairs=_as_int(env, "ROOSTOO_MAX_PAIRS", 8),
            max_spread_bps=_as_float(env, "MAX_SPREAD_BPS", 10.0),
            depth_provider=(_get(env, "DEPTH_PROVIDER", "none") or "none").strip().lower(),
            depth_path=_get(env, "ROOSTOO_DEPTH_PATH", "") or "",
            depth_band_pct=_as_float(env, "DEPTH_BAND_PCT", 0.005),
            depth_target_notional=_as_float(env, "DEPTH_TARGET_NOTIONAL", 2_000.0),
            depth_liquidity_share=_as_float(env, "DEPTH_LIQUIDITY_SHARE", 0.02),
            bar_seconds=_as_int(env, "BAR_SECONDS", 1800),
            loop_interval_sec=_as_float(env, "LOOP_INTERVAL_SEC", 60.0),
            history_window=_as_int(env, "HISTORY_WINDOW", 500),
            request_timeout_sec=_as_float(env, "REQUEST_TIMEOUT_SEC", 15.0),
            max_retries=_as_int(env, "MAX_RETRIES", 3),
            min_request_interval_sec=_as_float(env, "MIN_REQUEST_INTERVAL_SEC", 0.25),
            initial_capital=_as_float(env, "INITIAL_CAPITAL", 100_000.0),
            max_gross_exposure=_as_float(env, "MAX_GROSS_EXPOSURE", 0.60),
            max_pair_weight=_as_float(env, "MAX_PAIR_WEIGHT", 0.15),
            max_open_positions=_as_int(env, "MAX_OPEN_POSITIONS", 4),
            min_order_notional=_as_float(env, "MIN_ORDER_NOTIONAL", 25.0),
            risk_per_trade_pct=_as_float(env, "RISK_PER_TRADE_PCT", 0.005),
            max_daily_loss_pct=_as_float(env, "MAX_DAILY_LOSS_PCT", 0.02),
            max_drawdown_pct=_as_float(env, "MAX_DRAWDOWN_PCT", 0.20),
            cooldown_bars=_as_int(env, "COOLDOWN_BARS", 2),
            trading_day_offset_hours=_as_int(env, "TRADING_DAY_OFFSET_HOURS", 8),
            vol_target_annual=_as_float(env, "VOL_TARGET_ANNUAL", 0.25),
            stop_atr_mult=_as_optional_float(env, "STOP_ATR_MULT", 1.5),
            max_hold_bars=_as_optional_int(env, "MAX_HOLD_BARS", 12),
            stop_loss_pct=_as_optional_float(env, "STOP_LOSS_PCT", None),
            take_profit_pct=_as_optional_float(env, "TAKE_PROFIT_PCT", None),
            trailing_stop_pct=_as_optional_float(env, "TRAILING_STOP_PCT", None),
            crash_filter_pct=_as_float(env, "CRASH_FILTER_PCT", 0.12),
            refuse_entry_without_stop=_as_bool(_get(env, "REFUSE_ENTRY_WITHOUT_STOP"), True),
            limit_entries=_as_bool(_get(env, "LIMIT_ENTRIES"), False),
            limit_entry_offset_bps=_as_float(env, "LIMIT_ENTRY_OFFSET_BPS", 0.0),
            limit_entry_timeout_bars=_as_int(env, "LIMIT_ENTRY_TIMEOUT_BARS", 1),
            taker_fee=_as_float(env, "TAKER_FEE", DEFAULT_TAKER_FEE),
            maker_fee=_as_float(env, "MAKER_FEE", DEFAULT_MAKER_FEE),
            slippage_bps=_as_float(env, "SLIPPAGE_BPS", 5.0),
            strategy=_get(env, "STRATEGY", DEFAULT_STRATEGY) or DEFAULT_STRATEGY,
            strategy_params=params,
            log_level=(_get(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
            journal_dir=_get(env, "JOURNAL_DIR", "journal") or "journal",
            log_dir=_get(env, "LOG_DIR", "logs") or "logs",
            periods_per_year=_as_int(env, "PERIODS_PER_YEAR", 365),
            risk_free_rate=_as_float(env, "RISK_FREE_RATE", 0.0),
        )
        cfg.validate(require_keys=require_keys)
        return cfg

    # ------------------------------------------------------------------
    def validate(self, require_keys: bool = True) -> None:
        if not self.mock and require_keys:
            missing = [
                name
                for name, value in (("ROOSTOO_API_KEY", self.api_key), ("ROOSTOO_SECRET_KEY", self.secret_key))
                if not value
            ]
            if missing:
                raise ConfigError(
                    "missing credentials: "
                    + ", ".join(missing)
                    + ". Copy .env.example to .env and fill them in, or set ROOSTOO_MOCK=1 "
                    "to develop against the built-in simulator."
                )
        if self.loop_interval_sec < 5:
            # The rulebook bans HFT and the exchange throttles request bursts.
            raise ConfigError("LOOP_INTERVAL_SEC below 5s is not allowed by the competition rules")
        if self.bar_seconds < 60 or self.bar_seconds > 86_400:
            raise ConfigError("BAR_SECONDS must be between 60 and 86400")
        if 86_400 % self.bar_seconds:
            raise ConfigError("BAR_SECONDS must divide evenly into 86400 (e.g. 300, 900, 1800, 3600)")
        if self.loop_interval_sec > self.bar_seconds:
            raise ConfigError("LOOP_INTERVAL_SEC larger than BAR_SECONDS would skip whole bars")
        if not 0.0 < self.max_gross_exposure <= 1.0:
            raise ConfigError("MAX_GROSS_EXPOSURE must be in (0, 1]")
        if not 0.0 < self.max_pair_weight <= 1.0:
            raise ConfigError("MAX_PAIR_WEIGHT must be in (0, 1]")
        if self.max_pair_weight * self.max_open_positions < self.max_gross_exposure - 1e-9:
            # Warn-worthy, not fatal: positions simply cannot fill the budget.
            pass
        if self.history_window < 30:
            raise ConfigError("HISTORY_WINDOW must be >= 30 for indicators to be meaningful")
        if self.max_open_positions < 1:
            raise ConfigError("MAX_OPEN_POSITIONS must be >= 1")
        if self.max_pairs < 1:
            raise ConfigError("ROOSTOO_MAX_PAIRS must be >= 1")
        if self.max_spread_bps <= 0:
            raise ConfigError("MAX_SPREAD_BPS must be > 0")
        if not 0.0 < self.depth_band_pct < 0.5:
            raise ConfigError("DEPTH_BAND_PCT must be in (0, 0.5)")
        if self.depth_provider not in ("none", "ticker", "binance", "roostoo"):
            raise ConfigError("DEPTH_PROVIDER must be one of: none, ticker, binance, roostoo")
        if self.depth_provider == "roostoo" and not self.depth_path:
            raise ConfigError("DEPTH_PROVIDER=roostoo requires ROOSTOO_DEPTH_PATH")
        if not 0.0 <= self.taker_fee < 0.05 or not 0.0 <= self.maker_fee < 0.05:
            raise ConfigError("fee rates look wrong (expected fractions like 0.001)")
        if not 0.0 < self.risk_per_trade_pct <= 0.05:
            raise ConfigError("RISK_PER_TRADE_PCT must be in (0, 0.05]")
        if self.vol_target_annual < 0:
            raise ConfigError("VOL_TARGET_ANNUAL must be >= 0 (0 disables targeting)")
        if self.cooldown_bars < 0:
            raise ConfigError("COOLDOWN_BARS must be >= 0")
        for name in ("max_daily_loss_pct", "max_drawdown_pct"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ConfigError(f"{name} must be in (0, 1]")

    # ------------------------------------------------------------------
    @property
    def state_scope(self) -> dict[str, str]:
        """Identifies which runtime wrote a persisted state file.

        ``mock`` and ``live`` share the default journal directory, so without this
        a simulated book can be adopted by a live process: simulated positions, a
        simulated drawdown high-water mark, or a simulated halt, all read as real.

        Mode alone is not enough to identify a venue -- the test venue and the
        competition venue are both "live" -- so the base URL is part of the
        identity. Two different live venues must not inherit each other's stops.
        """
        return {"mode": "mock" if self.mock else "live", "venue": self.base_url}

    @property
    def state_dir_name(self) -> str:
        """Directory *inside* ``JOURNAL_DIR`` holding this runtime's state files.

        A stamp alone is not isolation. ``mock`` and ``live`` share
        ``JOURNAL_DIR`` by default, and a file whose stamp did not match used to be
        refusable for *reading* while remaining perfectly overwritable -- so a mock
        run replaced the live book, and the live cost basis and stop levels were
        gone for good. One directory per mode means a runtime never opens another
        runtime's state in the first place. The venue distinction is still carried
        by the stamp, which now also blocks writes.
        """
        return self.state_scope["mode"]

    def resolved_pairs(self) -> tuple[str, ...]:
        """Explicit universe, normalised to ``BASE/QUOTE`` form."""
        out = []
        for pair in self.pairs:
            normalised = pair.strip().upper()
            if "/" not in normalised:
                normalised = f"{normalised}/USD"
            out.append(normalised)
        return tuple(dict.fromkeys(out))  # de-duplicate, keep order

    def fee_for(self, order_type: str) -> float:
        return self.maker_fee if order_type.upper() == "LIMIT" else self.taker_fee

    def lot_for(self, pair: str) -> float:
        """Smallest tradable quantity increment for ``pair``.

        ``exchange_info`` is the only source of ``AmountPrecision``, so this reads
        it from the live venue description if the engine has one. The 1e-9
        fallback keeps callers total before the venue has been contacted.
        """
        resolver = getattr(self, "_lot_resolver", None)
        if callable(resolver):
            try:
                lot = float(resolver(pair))
                if lot > 0:
                    return lot
            except Exception:
                pass
        return 1e-9

    def bars_per_day(self) -> float:
        return 86_400.0 / float(self.bar_seconds)

    def ensure_dirs(self) -> None:
        for path in (self.journal_dir, self.log_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, Any]:
        """Config dump that is safe to print into logs."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("api_key", "secret_key"):
                value = f"***{str(value)[-4:]}" if value else ""
            out[f.name] = value
        return out
