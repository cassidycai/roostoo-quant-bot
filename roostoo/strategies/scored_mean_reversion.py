"""100-point, long-only mean reversion; existing mean exits are inherited."""

from __future__ import annotations

import math
from typing import Any

from .. import indicators as ind
from ..config import Config
from ..risk import PositionSizer
from .base import ENTER_LONG, MarketContext, Signal
from .mean_reversion import MeanReversionStrategy
from .scoring import SCORING_VERSION, economics, factor_scores


class ScoredMeanReversionStrategy(MeanReversionStrategy):
    name = "scored_mean_reversion"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "window": 48, "atr_period": 14, "adx_period": 14,
            "z_exit_long": -0.25, "z_exit_short": 0.25,
            "warmup_bars": 120, "score_threshold": 70.0,
            "min_deviation_score": 18, "min_reversal_score": 15,
            "z_min": -4.0, "z_max": -1.5, "adx_max": 30.0,
            "shock_max": 4.0, "expansion_max": 2.5,
            "shock_window": 48, "expansion_window": 48,
            "vwap_window": 24, "volume_window": 20,
            "volume_factors_enabled": False,
            "min_cost_coverage": 3.0, "min_net_rr": 1.2,
        }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._cfg: Config | None = None
        for key in ("window", "atr_period", "adx_period", "warmup_bars", "shock_window", "expansion_window", "vwap_window", "volume_window"):
            value = self.params[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{key} must be an integer >= 2")
        if not isinstance(self.params["volume_factors_enabled"], bool):
            raise ValueError("volume_factors_enabled must be a boolean")
        for key, value in self.params.items():
            if key == "volume_factors_enabled":
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
        if not 0 <= self.params["score_threshold"] <= 100:
            raise ValueError("score_threshold must be in [0, 100]")
        for key, cap in (("min_deviation_score", 30), ("min_reversal_score", 25)):
            if not 0 <= self.params[key] <= cap:
                raise ValueError(f"{key} outside category cap")
        if not self.params["z_min"] < self.params["z_max"] < self.params["z_exit_long"]:
            raise ValueError("require z_min < z_max < z_exit_long")
        for key in ("adx_max", "shock_max", "expansion_max", "min_cost_coverage", "min_net_rr"):
            if self.params[key] <= 0:
                raise ValueError(f"{key} must be positive")

    def configure_execution(self, cfg: Config) -> None:
        """Optional hook called by both runners; Strategy interface is unchanged."""
        if cfg.history_window < self.max_context_bars:
            raise ValueError("HISTORY_WINDOW too small for the scored strategy")
        if not (cfg.stop_atr_mult or cfg.stop_loss_pct):
            raise ValueError("scored strategy requires an ATR or percentage stop")
        self._cfg = cfg

    @property
    def min_bars(self) -> int:
        return self._required_bars()

    def _required_bars(self) -> int:
        p = self.params
        return max(p["warmup_bars"], p["window"] + 4,
                   2 * p["adx_period"] + 2, p["shock_window"] + 2,
                   p["atr_period"] + p["expansion_window"] + 1,
                   p["vwap_window"], p["volume_window"] + 1)

    def _entries(self, ctx: MarketContext, need: int) -> list[Signal]:
        if self._cfg is None:
            raise RuntimeError("call configure_execution(cfg) before generating scored entries")
        cfg, p = self._cfg, self.params
        out: list[Signal] = []
        held = ctx.held_pairs()
        for pair in ctx.universe:
            if ctx.daily_halt or pair in held or pair in ctx.blocked:
                self.note(pair, entry_skipped="held, blocked or daily halt")
                continue
            bars = ctx.series(pair)
            if len(bars) < need:
                self.note(pair, entry_skipped=f"need {need} closed bars")
                continue
            bars = bars[-self.max_context_bars:]
            step = ctx.bar_seconds * 1000
            valid = all(
                all(math.isfinite(x) and x > 0 for x in (b.open, b.high, b.low, b.close))
                and b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high
                and b.ts_ms % step == 0
                for b in bars
            ) and all(b.ts_ms - a.ts_ms == step for a, b in zip(bars, bars[1:]))
            closed_ms = bars[-1].ts_ms + step
            if not valid or not 0 <= ctx.now_ms - closed_ms < step:
                self.note(pair, entry_skipped="invalid, missing, unclosed or stale bars")
                continue
            ticker = ctx.tickers.get(pair)
            if ticker is None or not all(math.isfinite(x) and x > 0 for x in (ticker.max_bid, ticker.min_ask)) or ticker.max_bid > ticker.min_ask:
                self.note(pair, entry_skipped="invalid bid/ask")
                continue
            if not ctx.is_backtest and not 0 <= ctx.now_ms - ticker.server_time_ms <= 2 * cfg.loop_interval_sec * 1000:
                self.note(pair, entry_skipped="stale quote")
                continue
            if ticker.spread_bps > cfg.max_spread_bps:
                self.note(pair, entry_skipped="spread exceeds configured limit")
                continue
            closes = [b.close for b in bars]
            highs, lows = [b.high for b in bars], [b.low for b in bars]
            zs = ind.zscore_series(closes, p["window"])
            atr = ind.atr_wilder(highs, lows, closes, p["atr_period"])
            adx = ind.adx(highs, lows, closes, p["adx_period"])
            returns = ind.log_returns(closes)
            sd_return = ind.stdev(returns[-p["shock_window"] - 1:-1])
            shock = returns[-1] / sd_return if sd_return else None
            # A Wilder pass gives ATR_t and the preceding ATR observations.
            atrs = [x / p["atr_period"] for x in ind._wilder(ind.true_range(highs, lows, closes), p["atr_period"])]
            baseline = ind.mean(atrs[-p["expansion_window"] - 1:-1])
            expansion = atr / baseline if atr and baseline else None
            mean = ind.sma(closes, p["window"])
            std = ind.stdev(closes[-p["window"]:])
            old_mean = ind.sma(closes[:-4], p["window"])
            essentials = (atr, adx, shock, expansion, mean, std, old_mean)
            if len(zs) < 2 or any(x is None or not math.isfinite(x) for x in essentials) or atr <= 0 or std <= 0:
                self.note(pair, entry_skipped="core indicator unavailable")
                continue
            z, dz = zs[-1], zs[-1] - zs[-2]
            self.note(pair, z=z, delta_z=dz, adx=adx, atr=atr, return_shock=shock, vol_expansion=expansion)
            if not p["z_min"] <= z <= p["z_max"] or adx >= p["adx_max"] or dz <= 0 or closes[-1] <= closes[-2] or abs(shock) >= p["shock_max"] or expansion >= p["expansion_max"]:
                self.note(pair, entry_skipped="deviation, reversal, trend or shock gate")
                continue
            volumes = [b.volume for b in bars]
            volume_ok = p["volume_factors_enabled"] and all(math.isfinite(v) and v >= 0 for v in volumes)
            vwap = ind.vwap(highs, lows, closes, volumes, p["vwap_window"]) if volume_ok else None
            rel = ind.relative_volume(volumes, p["volume_window"]) if volume_ok else None
            b = bars[-1]
            factors = factor_scores(z=z, delta_z=dz, adx=adx, atr=atr,
                                    open_price=b.open, high=b.high, low=b.low, close=b.close,
                                    sma_change=mean - old_mean, shock=shock, expansion=expansion,
                                    vwap=vwap, relative_volume=rel)
            target = mean + p["z_exit_long"] * std
            reference = ticker.mid
            distance = PositionSizer(cfg).stop_distance(reference, atr)
            cost = 2 * cfg.taker_fee + 2 * cfg.slippage_bps / 10_000 + ticker.spread_bps / 10_000
            try:
                econ = economics(target, reference, distance if distance is not None else 0, cost)
            except ValueError:
                self.note(pair, entry_skipped="invalid stop or economics")
                continue
            factors["cost"] = int(econ["cost_score"])
            score = sum(factors.values())
            self.note(pair, entry_score=score, factor_scores=factors, economics=econ, volume_factors_used=bool(volume_ok), vwap=vwap, relative_volume=rel)
            if factors["deviation"] < p["min_deviation_score"] or factors["reversal"] < p["min_reversal_score"]:
                self.note(pair, entry_skipped="core category score too low")
                continue
            if econ["cost_coverage"] < p["min_cost_coverage"] or econ["net_rr"] < p["min_net_rr"]:
                self.note(pair, entry_skipped="insufficient cost coverage or net reward/risk")
                continue
            if score < p["score_threshold"]:
                self.note(pair, entry_skipped="total score below threshold")
                continue
            meta = self._meta(pair, z, zs[-2], adx, atr, bars)
            meta["atr"] = atr  # Preserve stop precision for low-price coins.
            meta.update(econ)
            meta.update({"scoring_version": SCORING_VERSION, "entry_score": score,
                         "factor_scores": factors, "score_without_cost": score - factors["cost"],
                         "target_price": target, "signal_close_ms": closed_ms,
                         "score_threshold": p["score_threshold"],
                         "min_cost_coverage": p["min_cost_coverage"], "min_net_rr": p["min_net_rr"],
                         "signal_id": f"{SCORING_VERSION}:{pair}:{b.ts_ms}"})
            out.append(Signal(pair, ENTER_LONG, strength=score / 100,
                              reason=f"entry score {score}/100 >= {p['score_threshold']:g}", meta=meta))
        return sorted(out, key=lambda s: (-s.meta["entry_score"], -s.meta["net_rr"], s.meta["cost_pct"], s.pair))
