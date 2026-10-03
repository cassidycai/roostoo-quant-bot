"""Score boundaries, signal gates, risk enforcement and delayed execution."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from roostoo.backtest import Backtester
from roostoo.candles import Candle
from roostoo.config import Config
from roostoo.models import Position, Ticker
from roostoo.risk import PortfolioView, RiskManager
from roostoo.strategies.base import ENTER_LONG, ENTER_SHORT, EXIT_LONG, MarketContext, Signal, load_strategy
from roostoo.strategies.mean_reversion import MeanReversionStrategy
from roostoo.strategies.scored_mean_reversion import ScoredMeanReversionStrategy
from roostoo.strategies.scoring import SCORING_VERSION, economics, execution_terms, factor_scores, risk_fraction

PAIR = "BTC/USD"
STEP = 1_800_000


def ticker(price=95.0, now=120 * STEP):
    return Ticker(PAIR, price, price * 0.99975, price * 1.00025, 0, 100, 10000, now)


def context():
    bars = [Candle(i * STEP, 100, 101, 99, 100, 10) for i in range(118)]
    bars += [Candle(118 * STEP, 95, 95.5, 93.5, 94, 10),
             Candle(119 * STEP, 94.8, 95.1, 94.1, 95, 15)]
    return MarketContext(120 * STEP, 1800, {PAIR: bars}, {PAIR: ticker()},
                         nav=100000, cash_usd=100000, universe=[PAIR], is_backtest=True)


def metadata(subtotal=75, threshold=70, target=105):
    return {"scoring_version": SCORING_VERSION, "signal_close_ms": 120 * STEP,
            "target_price": target, "score_without_cost": subtotal,
            "entry_score": subtotal + 10, "score_threshold": threshold,
            "min_cost_coverage": 3, "min_net_rr": 1.2, "atr": 1,
            "factor_scores": {"cost": 10}}


class FactorTests(unittest.TestCase):
    def scores(self, **overrides):
        args = dict(z=-2.6, delta_z=0.4, adx=15, atr=1,
                    open_price=94.8, high=95.1, low=94.1, close=95,
                    sma_change=0, shock=0.5, expansion=1, vwap=96, relative_volume=1.5)
        args.update(overrides)
        return factor_scores(**args)

    def test_maximum_and_no_volume_reweighting(self):
        self.assertEqual(self.scores(), {"deviation": 30, "reversal": 25, "regime": 20, "safety": 15})
        self.assertEqual(sum(self.scores(vwap=None, relative_volume=None).values()), 80)

    def test_exclusive_z_boundaries(self):
        for z, expected in [(-1.5, 8), (-2, 18), (-2.5, 25), (-3, 25), (-3.01, 12), (-4, 12), (-4.01, 0)]:
            with self.subTest(z=z):
                self.assertEqual(self.scores(z=z, vwap=None)["deviation"], expected)

    def test_stronger_shocks_and_extreme_volume_do_not_add_points(self):
        self.assertLess(self.scores(shock=3)["safety"], self.scores()["safety"])
        self.assertEqual(self.scores(relative_volume=3)["safety"], 10)
        self.assertEqual(self.scores(close=94.7)["safety"], 10)

    def test_cost_score_and_invalid_inputs(self):
        self.assertEqual(economics(106, 100, 2, 0.01)["cost_score"], 10)
        self.assertEqual(economics(104.5, 100, 2, 0.01)["cost_score"], 7)
        self.assertEqual(economics(103.5, 100, 2, 0.01)["cost_score"], 4)
        for bad in [0, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                economics(106, 100, 2, bad)

    def test_risk_tiers(self):
        self.assertEqual([risk_fraction(x) for x in (70, 79, 80, 89, 90)], [.002, .002, .0035, .0035, .005])


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.strategy = ScoredMeanReversionStrategy()
        self.strategy.configure_execution(Config())
        self.ctx = context()

    def generate(self, **indicators):
        # Separate the scoring/gating contract from Wilder indicator tests,
        # which already cover the repository's indicator implementations.
        defaults = {"zscore_series": [-2.9, -2.6], "atr_wilder": 1.0,
                    "adx": 15.0, "log_returns": [0.01, -0.01] * 60,
                    "stdev": 2.0, "sma": 100.0, "_wilder": [14.0] * 100}
        defaults.update(indicators)
        patches = [patch(f"roostoo.strategies.scored_mean_reversion.ind.{key}", return_value=value)
                   for key, value in defaults.items()]
        for item in patches:
            item.start()
        try:
            return self.strategy.generate(self.ctx)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_valid_entry_metadata(self):
        signals = self.generate()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, ENTER_LONG)
        self.assertEqual(signals[0].meta["entry_score"], 85)
        self.assertEqual(signals[0].meta["factor_scores"]["deviation"], 25)
        self.assertFalse(self.strategy.diagnostics[PAIR]["volume_factors_used"])

    def test_threshold_equality_then_rejection(self):
        self.strategy.params["score_threshold"] = 85
        self.assertEqual(len(self.generate()), 1)
        self.strategy.params["score_threshold"] = 86
        self.assertEqual(self.generate(), [])

    def test_core_minimum_cannot_be_bypassed_by_low_total_threshold(self):
        self.strategy.params["score_threshold"] = 0
        self.assertEqual(self.generate(zscore_series=[-1.9, -1.8]), [])

    def test_hard_gates(self):
        for overrides in ({"adx": 30}, {"zscore_series": [-4.3, -4.1]},
                          {"zscore_series": [-2.5, -2.6]}, {"atr_wilder": math.nan}):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.generate(**overrides), [])

    def test_price_must_rise_even_if_z_improves(self):
        self.ctx.candles[PAIR][-1] = replace(self.ctx.candles[PAIR][-1], close=94.0)
        self.assertEqual(self.generate(), [])

    def test_missing_gap_future_and_unclosed_bars(self):
        for mode in ("short", "gap", "future", "nan"):
            self.ctx = context()
            if mode == "short":
                self.ctx.candles[PAIR] = self.ctx.candles[PAIR][-20:]
            elif mode == "gap":
                self.ctx.candles[PAIR][30] = replace(self.ctx.candles[PAIR][30], ts_ms=29 * STEP)
            elif mode == "future":
                self.ctx.now_ms -= 1
            else:
                self.ctx.candles[PAIR][30] = replace(self.ctx.candles[PAIR][30], close=math.nan)
            with self.subTest(mode=mode):
                self.assertEqual(self.generate(), [])

    def test_stale_quote_in_live_mode(self):
        self.ctx.is_backtest = False
        self.ctx.tickers[PAIR] = replace(ticker(), server_time_ms=self.ctx.now_ms - 121000)
        self.assertEqual(self.generate(), [])

    def test_daily_halt_blocked_and_held(self):
        for mode in ("halt", "blocked", "held"):
            self.ctx = context()
            if mode == "halt":
                self.ctx.daily_halt = True
            elif mode == "blocked":
                self.ctx.blocked.add(PAIR)
            else:
                self.ctx.positions[PAIR] = Position(PAIR, quantity=1)
            self.assertFalse(any(s.is_entry for s in self.generate()))

    def test_mean_exit_does_not_depend_on_entry_score(self):
        self.ctx.positions[PAIR] = Position(PAIR, quantity=1)
        self.strategy.params["score_threshold"] = 100
        signals = self.generate(zscore_series=[-0.3, -0.2])
        self.assertEqual([s.action for s in signals], [EXIT_LONG])

    def test_deterministic_ranking(self):
        self.ctx.candles["AAA/USD"] = list(self.ctx.candles[PAIR])
        self.ctx.tickers["AAA/USD"] = replace(ticker(), pair="AAA/USD")
        self.ctx.universe.append("AAA/USD")
        self.assertEqual([s.pair for s in self.generate()], ["AAA/USD", PAIR])

    def test_parameter_validation_and_config_binding(self):
        for params in ({"score_threshold": math.nan}, {"window": 1}, {"volume_factors_enabled": "false"}, {"z_entry": 2}):
            with self.assertRaises(ValueError):
                ScoredMeanReversionStrategy(params)
        with self.assertRaises(ValueError):
            self.strategy.configure_execution(Config(stop_atr_mult=None, stop_loss_pct=None))
        with self.assertRaises(ValueError):
            self.strategy.configure_execution(Config(history_window=50))
        loaded = load_strategy("roostoo.strategies.scored_mean_reversion:ScoredMeanReversionStrategy")
        self.assertIsInstance(loaded, ScoredMeanReversionStrategy)


class RiskAndFillTests(unittest.TestCase):
    def approve(self, meta=None, cfg=None, **kwargs):
        cfg = cfg or Config()
        meta = metadata() if meta is None else meta
        manager = RiskManager(cfg)
        signal = Signal(PAIR, ENTER_LONG, meta=meta)
        return manager.evaluate([signal], view=PortfolioView(100000, 100000),
                                tickers={PAIR: ticker(100)}, now_ms=120 * STEP,
                                bar_idx=120, **kwargs)

    def test_risk_budget_includes_cost_and_cfg_ceiling(self):
        action = self.approve(cfg=Config(max_pair_weight=.4, max_gross_exposure=.8, max_open_positions=2)).approved[0]
        loss_fraction = (100 - action.stop_price) / 100 + action.meta["cost_pct"]
        self.assertAlmostEqual(action.notional * loss_fraction, 350)
        self.assertAlmostEqual(action.risk_amount, 350)
        lower = self.approve(cfg=Config(risk_per_trade_pct=.001)).approved[0]
        self.assertAlmostEqual(lower.risk_amount, 100)

    def test_actual_costs_reject_entry(self):
        self.assertFalse(self.approve(cfg=Config(taker_fee=.01)).approved)

    def test_inflight_reservation_still_blocks_duplicate(self):
        self.assertFalse(self.approve(committed_pairs={PAIR}, committed_notional=1000).approved)

    def test_no_stop_and_malformed_metadata_fail_closed(self):
        self.assertFalse(self.approve(cfg=Config(stop_atr_mult=None)).approved)
        self.assertFalse(self.approve(meta={"scoring_version": SCORING_VERSION}).approved)
        self.assertFalse(self.approve(meta={**metadata(), "target_price": math.nan}).approved)

    def test_short_scored_signal_rejected(self):
        manager = RiskManager(Config())
        result = manager.evaluate([Signal(PAIR, ENTER_SHORT, meta=metadata())],
                                  view=PortfolioView(100000, 100000), tickers={PAIR: ticker(100)},
                                  now_ms=120 * STEP, bar_idx=120)
        self.assertFalse(result.approved)

    def test_legacy_sizing_stays_unchanged(self):
        action = self.approve(meta={"atr": 1}).approved[0]
        self.assertEqual(action.notional, 15000)
        self.assertEqual(action.risk_amount, 500)

    def test_signal_expiry(self):
        error, _ = execution_terms(Config(), metadata(), reference=100,
                                   spread_bps=5, stop_price=98.5, now_ms=121 * STEP)
        self.assertIn("expired", error)

    def test_fill_rechecks_next_open_and_gap_risk(self):
        cfg = Config()
        bt = Backtester(cfg, MeanReversionStrategy(), {PAIR: context().candles[PAIR]})
        rejected = self.approve().approved[0]
        bt._fill(rejected, reference_price=104.5, ts=120 * STEP, source="open")
        self.assertEqual(bt.trades, [])
        action = self.approve(meta=metadata(target=110)).approved[0]
        original_quantity = action.quantity
        bt._fill(action, reference_price=102, ts=120 * STEP, source="open")
        self.assertEqual(len(bt.trades), 1)
        self.assertLess(action.quantity, original_quantity)
        self.assertLessEqual(action.risk_amount, 350.000001)

    def test_expired_delayed_fill_cannot_trade(self):
        bt = Backtester(Config(), MeanReversionStrategy(), {PAIR: context().candles[PAIR]})
        bt._fill(self.approve().approved[0], reference_price=100, ts=121 * STEP, source="open")
        self.assertEqual(bt.trades, [])

    def test_backtest_binds_real_configuration(self):
        strategy = ScoredMeanReversionStrategy()
        cfg = Config(taker_fee=.002)
        Backtester(cfg, strategy, {PAIR: context().candles[PAIR]})
        self.assertIs(strategy._cfg, cfg)


if __name__ == "__main__":
    unittest.main()
