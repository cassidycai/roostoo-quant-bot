"""Rules 2 and 3 -- the 30-minute mean-reversion core.

Rule 2, restated precisely::

    SMA48 = mean of the last 48 closes            (48 x 30min = 24h)
    Std48 = standard deviation of those closes
    Z     = (Close - SMA48) / Std48

    LONG  entry:  Z[t-1] <= -X  AND  Z[t] > Z[t-1]     (stretched down, turning up)
    SHORT entry:  Z[t-1] >= +X  AND  Z[t] < Z[t-1]     (stretched up, turning down)
    Exit long  when Z >= -0.25 ; exit short when Z <= +0.25

Rule 3: only take those entries when ``ADX(14) < 25``.

``direction`` selects which side of the mean we buy, and it moves the EXITS with
the entries -- the two must agree, or the Z-exit silently never fires and every
position dies on its stop instead:

  ``momentum`` (default)   long a stretch UP,    exit when Z returns to -0.25
                           short a stretch DOWN,  exit when Z returns to +0.25
  ``reversion``            long a stretch DOWN,  exit when Z returns to -0.25
                           short a stretch UP,    exit when Z returns to +0.25

The rulebook text for the long clause reads *"Z[t-1] >= X and Delta Z < 0"*,
which is the fade-a-rally setup, so its *level* matches ``momentum`` and its
*confirmation sign* matches ``reversion``; the text is ambiguous and the team
resolved it in favour of momentum, measured (docs/FINDINGS.md section 9).

Two notes on the specification as written.

1. The team's text says *"Long Entry when Z[t-1] >= X and Delta Z < 0"* for the
   second clause. That is the **short** condition -- a long entry needs the price
   below the mean, and ``Z >= +2`` is the opposite. Implemented symmetrically
   here, with ``allow_short`` to switch the short leg off.

2. "``Delta Z > 0``" is a *turn* confirmation, not just a level. It is what stops
   the bot catching a falling knife: ``Z = -2.4`` alone is a knife; ``Z`` going
   from ``-2.4`` to ``-2.1`` means the stretch is already contracting. The
   previous bar is therefore structurally required, which is why ``min_bars`` is
   ``window + 1`` and not ``window``.

Both entry conditions and both filter conditions are recorded per pair per bar in
``self.diagnostics``, so a backtest can answer "why were there only three trades?"
without re-instrumenting anything.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import indicators as ind
from ..candles import Candle
from .base import (
    ENTER_LONG,
    ENTER_SHORT,
    EXIT_LONG,
    EXIT_SHORT,
    MarketContext,
    Signal,
    Strategy,
)


class MeanReversionStrategy(Strategy):
    """Z-score reversion on 30-minute bars, gated by ADX."""

    name = "mean_reversion"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # --- Rule 2 -----------------------------------------------------
            "window": 48,  # 48 bars of 30min = 24h
            "z_entry": 2.0,  # the "X" of Rule 2
            "z_exit_long": -0.25,
            "z_exit_short": 0.25,
            "require_turn": True,  # Delta Z confirmation
            #: Minimum |dZ| for the turn confirmation. 0.0 reproduces the
            #: historical "any move in the entry's direction" test.
            "min_delta_z": 0.0,
            #: "momentum" (the shipped policy) rides the deviation; "reversion"
            #: fades it. Both the entries and the exits follow this switch, and
            #: they must agree or the Z-exit never fires.
            #: 8 pairs x 120 days, costs and stops included (FINDINGS section 9):
            #:   reversion  IS -6.06%  OOS -4.22%  dd 6.28%  104 round trips
            #:   momentum   IS -2.20%  OOS -0.55%  dd 2.50%   37 round trips
            "direction": "momentum",
            # --- Rule 3 -----------------------------------------------------
            "adx_period": 14,
            "adx_max": 25.0,
            "require_adx": True,  # False = trade without the trend filter (A/B test)
            # --- Rule 4 (owned by a teammate -- OFF by default) --------------
            # This gate belongs to Rule 4, which is outside this module's scope, so
            # it ships disabled: Rules 1-3 must not silently execute someone
            # else's rule. It is worth measuring, though -- Rule 2 on its own does
            # fire on economically trivial moves (in a quiet market a 2-sigma
            # deviation can be smaller than the 0.2% round-trip commission), and
            # docs/FINDINGS.md section 4 shows it helping at every z-entry level.
            # Enable with: STRATEGY_PARAMS='{"enforce_min_deviation": true}'
            "min_abs_deviation": 0.006,  # |Close - SMA48| / Close > 0.6%
            "enforce_min_deviation": False,
            # --- Rule 5 inputs (teammate-owned) -----------------------------
            "atr_period": 14,  # ATR is reported in meta; the stop itself is risk.py's job
            # --- direction --------------------------------------------------
            "allow_short": True,
            # --- supplementary rules (teammate-owned; math in indicators.py)
            # All default OFF, per the team's plan to add them one at a time and
            # keep whichever improves the backtest.
            "enable_return_shock_filter": False,
            "return_shock_max": 2.5,
            "enable_vwap_filter": False,
            "enable_relative_volume_filter": False,
            "relative_volume_min": 1.2,
            "relative_volume_max": 2.5,
            "enable_vol_expansion_filter": False,
            "vol_expansion_max": 2.5,
        }

    # ------------------------------------------------------------------
    @property
    def min_bars(self) -> int:
        """Bars needed for ``Z[t-1]`` *and* ``Z[t]``, plus a warm-up margin."""
        return int(self.params["window"]) + 2

    @property
    def required_bars(self) -> int:
        return self._required_bars()

    @property
    def max_context_bars(self) -> int:
        """Just enough history for every enabled indicator, plus a small margin.

        Rule 2 needs ``Z[t-1]`` as well as ``Z[t]``, so the margin keeps the
        lookback from starving. Without a bound here the engine hands over its
        whole rolling buffer and each indicator rescans it every bar -- the
        dominant cost of a year-long, multi-pair backtest.
        """
        return self._required_bars() + 5

    def _required_bars(self) -> int:
        """Longest lookback any enabled filter needs."""
        need = int(self.params["window"]) + 2
        if self.params["require_adx"] or self.params["enable_return_shock_filter"]:
            need = max(need, 2 * int(self.params["adx_period"]) + 2)
        if self.params["enable_vol_expansion_filter"]:
            need = max(need, int(self.params["window"]) + 2)
        return need

    # ------------------------------------------------------------------
    def generate(self, ctx: MarketContext) -> list[Signal]:
        self.reset_diagnostics()
        signals: list[Signal] = []
        need = self._required_bars()

        # Exits first: freeing a slot this bar is always preferable to wanting one.
        signals.extend(self._exits(ctx, need))
        signals.extend(self._entries(ctx, need))
        return signals

    def _momentum(self) -> bool:
        """True when the deviation is taken as continuation rather than a fade."""
        return str(self.params.get("direction", "momentum")).strip().lower() != "reversion"

    # ------------------------------------------------------------------
    def _exits(self, ctx: MarketContext, need: int) -> list[Signal]:
        out: list[Signal] = []
        window = int(self.params["window"])
        for pair, position in ctx.positions.items():
            if not (position.quantity > 0 or position.is_short):
                continue
            closes = ctx.closes(pair)
            if len(closes) < window:
                self.note(pair, exit_skipped="insufficient history")
                continue
            z_series = ind.zscore_series(closes, window)
            if not z_series:
                continue
            z = z_series[-1]
            series = ctx.series(pair)
            atr_value = ind.atr_wilder(ctx.highs(pair), ctx.lows(pair), closes, int(self.params["atr_period"]))

            momentum = self._momentum()
            if position.is_short:
                exit_now = (z >= float(self.params["z_exit_short"]) if momentum
                            else z <= float(self.params["z_exit_short"]))
                if exit_now:
                    out.append(
                        Signal(
                            pair,
                            EXIT_SHORT,
                            reason=(
                                f"Z {z:+.2f} {'>=' if momentum else '<='} "
                                f"{self.params['z_exit_short']:+.2f} (mean reached)"
                            ),
                            meta=self._meta(pair, z, None, None, atr_value, series),
                        )
                    )
                else:
                    self.note(pair, short_exit_held_for=f"Z {z:+.2f}")
            else:
                exit_now = (z <= float(self.params["z_exit_long"]) if momentum
                            else z >= float(self.params["z_exit_long"]))
                if exit_now:
                    out.append(
                        Signal(
                            pair,
                            EXIT_LONG,
                            reason=(
                                f"Z {z:+.2f} {'<=' if momentum else '>='} "
                                f"{self.params['z_exit_long']:+.2f} (mean reached)"
                            ),
                            meta=self._meta(pair, z, None, None, atr_value, series),
                        )
                    )
                else:
                    self.note(pair, long_exit_held_for=f"Z {z:+.2f}")
        return out

    # ------------------------------------------------------------------
    def _entries(self, ctx: MarketContext, need: int) -> list[Signal]:
        out: list[Signal] = []
        window = int(self.params["window"])
        z_entry = float(self.params["z_entry"])
        held = ctx.held_pairs()

        for pair in ctx.universe:
            if pair in held:
                self.note(pair, entry_skipped="already holding")
                continue
            if pair in ctx.blocked:
                self.note(pair, entry_skipped="blocked (cooldown or daily halt)")
                continue
            if not ctx.has_bars(pair, need):
                self.note(pair, entry_skipped=f"need {need} bars, have {ctx.bar_count(pair)}")
                continue

            closes = ctx.closes(pair)
            z_series = ind.zscore_series(closes, window)
            if len(z_series) < 2:
                self.note(pair, entry_skipped="z-score series too short")
                continue
            z_now, z_prev = z_series[-1], z_series[-2]
            delta_z = z_now - z_prev
            highs, lows = ctx.highs(pair), ctx.lows(pair)
            atr_value = ind.atr_wilder(highs, lows, closes, int(self.params["atr_period"]))
            adx_value = ind.adx(highs, lows, closes, int(self.params["adx_period"]))
            deviation = ind.price_deviation_pct(closes, window)

            self.note(
                pair,
                z_now=round(z_now, 3),
                z_prev=round(z_prev, 3),
                delta_z=round(delta_z, 4),
                adx=None if adx_value is None else round(adx_value, 2),
                deviation=None if deviation is None else round(deviation, 5),
                atr=None if atr_value is None else round(atr_value, 6),
            )

            # --- Rule 3: refuse to fade a trend -------------------------
            if self.params["require_adx"]:
                if adx_value is None:
                    self.note(pair, entry_skipped="ADX unavailable")
                    continue
                if adx_value >= float(self.params["adx_max"]):
                    self.note(pair, entry_skipped=f"ADX {adx_value:.1f} >= {self.params['adx_max']:.1f} (trending)")
                    continue

            # --- Rule 4: the move must be worth the fees ---------------
            if self.params["enforce_min_deviation"]:
                if deviation is None:
                    self.note(pair, entry_skipped="deviation unavailable")
                    continue
                if deviation <= float(self.params["min_abs_deviation"]):
                    self.note(
                        pair,
                        entry_skipped=(
                            f"|dev| {deviation * 100:.2f}% <= {self.params['min_abs_deviation'] * 100:.2f}%"
                        ),
                    )
                    continue

            # --- supplementary filters (all off by default) ------------
            if not self._supplementary_ok(ctx, pair, closes):
                continue

            # --- Rule 2: the entry itself ------------------------------
            # `direction` chooses which side of the mean we buy. `min_delta_z`
            # is the turn confirmation: the Z move must be at least this large
            # in the entry's direction. 0.0 reproduces the historical test.
            if self._momentum():
                long_trigger = z_prev >= z_entry
                short_trigger = z_prev <= -z_entry
            else:
                long_trigger = z_prev <= -z_entry
                short_trigger = z_prev >= z_entry
            min_dz = float(self.params.get("min_delta_z", 0.0) or 0.0)
            if self.params["require_turn"]:
                long_trigger = long_trigger and delta_z > min_dz
                short_trigger = short_trigger and delta_z < -min_dz

            meta = self._meta(pair, z_now, z_prev, adx_value, atr_value, ctx.series(pair))
            meta["deviation"] = deviation
            if long_trigger:
                out.append(
                    Signal(
                        pair,
                        ENTER_LONG,
                        strength=self._strength(z_prev, z_entry),
                        reason=(
                            f"Z {z_prev:+.2f} <= {-z_entry:+.2f} and turning (dZ {delta_z:+.3f}); "
                            f"ADX {adx_value:.1f}" if adx_value is not None else f"Z {z_prev:+.2f} <= {-z_entry:+.2f}"
                        ),
                        meta=meta,
                    )
                )
            elif short_trigger and self.params["allow_short"]:
                out.append(
                    Signal(
                        pair,
                        ENTER_SHORT,
                        strength=self._strength(z_prev, z_entry),
                        reason=(
                            f"Z {z_prev:+.2f} >= {z_entry:+.2f} and turning (dZ {delta_z:+.3f}); "
                            f"ADX {adx_value:.1f}" if adx_value is not None else f"Z {z_prev:+.2f} >= {z_entry:+.2f}"
                        ),
                        meta=meta,
                    )
                )
            else:
                self.note(pair, entry_skipped=f"no trigger (Z {z_prev:+.2f} -> {z_now:+.2f})")
        return out

    # ------------------------------------------------------------------
    def _supplementary_ok(self, ctx: MarketContext, pair: str, closes: list[float]) -> bool:
        """The four supplementary filters, each independently switchable."""
        highs, lows, volumes = ctx.highs(pair), ctx.lows(pair), ctx.volumes(pair)

        if self.params["enable_return_shock_filter"]:
            shock = ind.return_shock_z(closes, 12)
            self.note(pair, return_shock_z=None if shock is None else round(shock, 3))
            if shock is None:
                self.note(pair, entry_skipped="return shock unavailable")
                return False
            if abs(shock) > float(self.params["return_shock_max"]):
                self.note(pair, entry_skipped=f"|return shock Z| {abs(shock):.2f} > {self.params['return_shock_max']}")
                return False

        if self.params["enable_vwap_filter"]:
            gap = ind.vwap_gap(highs, lows, closes, volumes, int(self.params["window"]))
            self.note(pair, vwap_gap=None if gap is None else round(gap, 5))
            if gap is None:
                self.note(pair, entry_skipped="VWAP unavailable")
                return False

        if self.params["enable_relative_volume_filter"]:
            rel = ind.relative_volume(volumes, int(self.params["window"]))
            self.note(pair, relative_volume=None if rel is None else round(rel, 3))
            if rel is None:
                self.note(pair, entry_skipped="relative volume unavailable")
                return False
            if not (float(self.params["relative_volume_min"]) <= rel <= float(self.params["relative_volume_max"])):
                self.note(
                    pair,
                    entry_skipped=(
                        f"rel volume {rel:.2f} outside "
                        f"[{self.params['relative_volume_min']}, {self.params['relative_volume_max']}]"
                    ),
                )
                return False

        if self.params["enable_vol_expansion_filter"]:
            expansion = ind.vol_expansion(highs, lows, 6, int(self.params["window"]))
            self.note(pair, vol_expansion=None if expansion is None else round(expansion, 3))
            if expansion is None:
                self.note(pair, entry_skipped="vol expansion unavailable")
                return False
            if expansion >= float(self.params["vol_expansion_max"]):
                self.note(
                    pair,
                    entry_skipped=f"vol expansion {expansion:.2f} >= {self.params['vol_expansion_max']}",
                )
                return False

        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _strength(z_prev: float, z_entry: float) -> float:
        """Scale conviction by how far past the trigger the deviation went.

        1.0 at the threshold, capped at 1.5 for a 50% overshoot. Available to the
        risk layer for scaling size; the default sizing ignores it.
        """
        if z_entry <= 0:
            return 1.0
        overshoot = abs(z_prev) / z_entry
        return max(0.0, min(1.5, overshoot))

    @staticmethod
    def _meta(
        pair: str,
        z: float,
        z_prev: Optional[float],
        adx_value: Optional[float],
        atr_value: Optional[float],
        series: list[Candle],
    ) -> dict[str, Any]:
        price = series[-1].close if series else 0.0
        return {
            "z": round(z, 4),
            "z_prev": None if z_prev is None else round(z_prev, 4),
            "adx": None if adx_value is None else round(adx_value, 2),
            "atr": None if atr_value is None else round(atr_value, 6),
            "atr_pct": None if (atr_value is None or price <= 0) else round(atr_value / price, 5),
            "close": round(price, 6),
            "bar_ts_ms": series[-1].ts_ms if series else 0,
        }
