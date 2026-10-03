#!/usr/bin/env python3
"""Run the trading bot.

    python run_live.py --check              # read-only venue verification (do this first)
    python run_live.py --mock --cycles 20   # full loop against the built-in simulator
    python run_live.py                      # live, forever

``--check`` is the tool for Oct 1-3: it proves the keys sign correctly and that
the clock, universe and balances all look sane, **without sending a single
order**. That matters, because the rulebook forbids manual API calls that trade,
and because a signature bug found on Oct 4 costs a day of the competition.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roostoo import config as config_mod  # noqa: E402
from roostoo.basis import BasisMonitor  # noqa: E402
from roostoo.client import build_client  # noqa: E402
from roostoo.engine import TradingEngine  # noqa: E402
from roostoo.journal import Journal  # noqa: E402
from roostoo.universe import build_depth_provider  # noqa: E402

log = logging.getLogger("run_live")

#: Exit status when the portfolio kill switch has fired. The state is persisted as
#: halted, so restarting cannot help -- the unit declares this status a clean stop
#: (`SuccessExitStatus`) instead of letting `Restart=always` thrash.
HALTED_EXIT_STATUS = 3


# ---------------------------------------------------------------------------
# Read-only venue check
# ---------------------------------------------------------------------------


def run_check(cfg) -> int:
    """Verify credentials, connectivity and the tradable universe. No orders."""
    client = build_client(cfg)
    problems: list[str] = []
    print("=" * 68)
    print("Roostoo connectivity check (read-only: no orders are sent)")
    print("=" * 68)

    try:
        offset = client.sync_time()
        print(f"server time      OK   clock offset {offset:+d} ms")
        if abs(offset) > 30_000:
            # Warn at half the limit: the client applies the offset, but a host
            # clock that far out is a symptom worth fixing before it drifts more.
            problems.append(
                f"host clock is {offset} ms from the exchange; the venue rejects "
                "signed requests beyond 60 s, and this is already half of that"
            )
    except Exception as exc:
        print(f"server time      FAIL {exc}")
        return 1

    try:
        info = client.exchange_info()
        tradable = [p for p, tp in info.pairs.items() if tp.can_trade]
        print(f"exchange info    OK   {len(info.pairs)} pairs, {len(tradable)} tradable, running={info.is_running}")
        print(f"initial wallet   {info.initial_wallet}")
        if not info.is_running:
            problems.append("exchange reports IsRunning=false")
        for pair in tradable[:12]:
            tp = info.pairs[pair]
            print(
                f"    {pair:<10} price_prec={tp.price_precision} amount_prec={tp.amount_precision} "
                f"min_order={tp.min_order:g}"
            )
    except Exception as exc:
        print(f"exchange info    FAIL {exc}")
        return 1

    tickers: dict = {}
    try:
        tickers = client.ticker()
        print(f"\ntickers          OK   {len(tickers)} pairs quoted")
        print(f"    {'pair':<10} {'last':>14} {'spread(bps)':>12} {'24h volume':>16}")
        ranked = sorted(tickers.values(), key=lambda t: t.unit_volume, reverse=True)
        for ticker in ranked[:12]:
            print(
                f"    {ticker.pair:<10} {ticker.last:>14,.6f} {ticker.spread_bps:>12.2f} "
                f"{ticker.unit_volume:>16,.0f}"
            )
        wide = [t.pair for t in ranked if t.spread_bps > cfg.max_spread_bps]
        if wide:
            print(f"    pairs above the {cfg.max_spread_bps:.1f}bps ceiling: {wide[:8]}")
    except Exception as exc:
        print(f"tickers          FAIL {exc}")
        problems.append("ticker endpoint failed")

    # The organisers confirmed the mock venue tracks Binance. Verify it once,
    # read-only: if the basis is wide, every Rules 2-3 signal is being computed
    # against a feed that has drifted, and the fix is the symbol mapping or the
    # feed, not the strategy.
    try:
        report = BasisMonitor(timeout=cfg.request_timeout_sec).check(tickers)
        if report.source_ok:
            print(f"\nbasis vs Binance  OK   tolerance +/-{report.threshold_pct * 100:.2f}%")
            print(f"    {'pair':<10} {'venue mid':>15} {'binance':>15} {'basis':>11}")
            for pair in sorted(report.rows):
                row = report.rows[pair]
                ref = "n/a" if row.reference_price is None else f"{row.reference_price:,.6f}"
                pct = "unverified" if row.basis_pct is None else f"{row.basis_pct * 100:+.3f}%"
                print(f"    {pair:<10} {row.venue_mid:>15,.6f} {ref:>15} {pct:>11}")
            if report.blocked():
                print(f"    OVER TOLERANCE: {sorted(report.blocked())}")
                problems.append(
                    f"basis wider than {report.threshold_pct * 100:.2f}% on {sorted(report.blocked())}: "
                    "check the symbol mapping (USDT vs USD) and feed freshness before trading"
                )
            if report.unverified:
                print(f"    no reference price for {sorted(report.unverified)} (not blocked)")
        else:
            print(f"\nbasis vs Binance  WARN {report.error}")
            print("    trading is still possible, but the venue-tracks-Binance premise is unverified")
    except Exception as exc:
        print(f"\nbasis vs Binance  WARN {exc}")

    # Rule 1's depth clause is pluggable and its failure mode depends on the
    # provider: `binance` FAILS CLOSED, so an unreachable depth endpoint excludes
    # every pair and the bot silently stops trading -- the worst possible thing
    # to discover during a scored window. README and AWS_DEPLOY both tell the
    # operator to confirm depth connectivity with `--check`, so actually exercise
    # the configured provider here instead of leaving that claim unbacked.
    if cfg.depth_provider != "none" and tickers:
        try:
            provider = build_depth_provider(cfg, tickers, client)
            probes = sorted(tickers.values(), key=lambda t: t.unit_volume, reverse=True)[: cfg.max_pairs]
            print(f"\ndepth ({provider.name})   OK   probing {len(probes)} pair(s), target ${cfg.depth_target_notional:,.0f}")
            usable = 0
            for ticker in probes:
                snapshot = provider.snapshot(ticker.pair, cfg.depth_band_pct)
                if snapshot is None:
                    print(f"    {ticker.pair:<10} unavailable")
                    continue
                available = snapshot.for_side("BUY")
                flag = "ok" if available >= cfg.depth_target_notional else "below target"
                print(f"    {ticker.pair:<10} ask-side within +/-{cfg.depth_band_pct * 100:.2f}%: {available:>14,.0f}  {flag}")
                if available >= cfg.depth_target_notional:
                    usable += 1
            if usable == 0:
                problems.append(
                    f"depth provider '{provider.name}' returned nothing usable for any probed pair. "
                    "It fails CLOSED, so with this setting Rule 1 rejects every pair and the bot will "
                    "not trade at all. Use DEPTH_PROVIDER=none until this check passes."
                )
        except Exception as exc:
            print(f"\ndepth            WARN {exc}")
            problems.append(f"depth provider '{cfg.depth_provider}' raised: {exc}")
    elif cfg.depth_provider == "none":
        print("\ndepth            SKIP DEPTH_PROVIDER=none (Rule 1 clause 3 is not enforced)")

    try:
        balances = client.balance()
        total_usd = balances["USD"].total if "USD" in balances else 0.0
        print(f"\nbalance          OK   USD total {total_usd:,.2f}")
        for asset, book in sorted(balances.items()):
            if book.total:
                print(f"    {asset:<8} free={book.free:,.8f} locked={book.locked:,.8f}")
    except Exception as exc:
        print(f"balance          FAIL {exc}")
        problems.append("balance endpoint failed (check the API key/signature)")

    try:
        total, by_pair = client.pending_count()
        print(f"pending orders   OK   {total} {by_pair if by_pair else ''}")
    except Exception as exc:
        print(f"pending orders   WARN {exc}")

    try:
        shorts = client.short_positions()
        print(f"short positions  OK   {len(shorts)} open")
    except Exception as exc:
        print(f"short positions  WARN {exc} (shorts may be disabled for this competition)")

    print("\n" + "-" * 68)
    if problems:
        print("RESULT: PROBLEMS FOUND")
        for problem in problems:
            print(f"  ! {problem}")
        return 2
    print("RESULT: all read-only checks passed. Signing and clock are good.")
    return 0


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------


def discover_seed_paths(data_dir: str, interval: str) -> dict[str, str]:
    """Every ``<COIN>-<UNIT>_<interval>.csv`` in ``data_dir``, keyed by pair.

    Scanning the directory instead of the configured pair list is deliberate.
    Rule 1 chooses the universe from 24h turnover on every bar, so the traded
    pairs are not known until the venue has been contacted; seeding only the
    configured ones both bypasses that ranking and silently starves the warm-up
    whenever the two disagree. That is what used to happen by default, because
    ``ROOSTOO_PAIRS`` ships empty.

    Handing over a broader set than necessary is safe: ``engine.seed_history``
    skips any pair the venue does not list and re-checks each file's last close
    against the live mid before using it.

    ``sample_*.csv`` files are synthetic fixtures for the tests, not history.
    """
    out: dict[str, str] = {}
    suffix = f"_{interval}.csv"
    for path in sorted(Path(data_dir).glob(f"*-*{suffix}")):
        if path.name.startswith("sample_"):
            continue
        stem = path.name[: -len(suffix)]
        coin, _, unit = stem.partition("-")
        if coin and unit:
            out[f"{coin.upper()}/{unit.upper()}"] = str(path)
    return out


def build_engine(cfg, seed: bool, data_dir: str, interval: str) -> TradingEngine:
    client = build_client(cfg)
    journal = Journal(cfg.journal_dir)
    seed_paths: dict[str, str] = {}
    if seed:
        seed_paths = discover_seed_paths(data_dir, interval)
    return TradingEngine(cfg, client=client, journal=journal, seed_paths=seed_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--check", action="store_true", help="read-only venue verification, then exit")
    parser.add_argument("--mock", action="store_true", help="use the built-in exchange simulator")
    parser.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = forever)")
    parser.add_argument("--flatten-on-exit", action="store_true", help="close every position on shutdown")
    parser.add_argument("--seed", dest="seed", action="store_true", default=True, help="warm up from data/ CSVs")
    parser.add_argument("--no-seed", dest="seed", action="store_false")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--interval", default="30m")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--params", default="", help="JSON object merged over the strategy defaults")
    parser.add_argument("--loop-interval", type=float, default=None, help="override LOOP_INTERVAL_SEC")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    config_mod.load_dotenv(args.env)
    try:
        cfg = config_mod.Config.from_env(require_keys=not args.mock)
        if args.mock:
            cfg.mock = True
            cfg.api_key = cfg.api_key or "MOCK"
            cfg.secret_key = cfg.secret_key or "MOCK"
        if args.strategy:
            cfg.strategy = args.strategy
        if args.params:
            import json

            cfg.strategy_params = {**cfg.strategy_params, **json.loads(args.params)}
        if args.loop_interval:
            cfg.loop_interval_sec = args.loop_interval
        cfg.validate(require_keys=not args.mock)
    except config_mod.ConfigError as exc:
        # A traceback here would be the first thing a teammate sees on a fresh
        # checkout, so say what is wrong and how to fix it instead.
        print(f"configuration error: {exc}", file=sys.stderr)
        print(
            "\nFix it in .env (copy .env.example first), or run fully offline:\n"
            "  python run_live.py --mock --cycles 20",
            file=sys.stderr,
        )
        return 2
    cfg.ensure_dirs()

    log_level = (args.log_level or cfg.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(cfg.log_dir) / "bot.log", encoding="utf-8"),
        ],
    )

    if args.check:
        return run_check(cfg)

    engine = build_engine(cfg, args.seed, args.data_dir, args.interval)
    log.info("strategy: %s", engine.strategy.describe())
    log.info("mode: %s, loop=%ss, bar=%ss", "MOCK" if cfg.mock else "LIVE", cfg.loop_interval_sec, cfg.bar_seconds)

    def handle_signal(signum, _frame):
        log.warning("received signal %s; shutting down%s", signum, " and flattening" if args.flatten_on_exit else "")
        engine._shutting_down = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, OSError):  # pragma: no cover - not all platforms
            pass

    stats = None
    try:
        stats = engine.run(max_cycles=args.cycles or None)
    finally:
        engine.shutdown(flatten=args.flatten_on_exit)
        if stats is not None:
            log.info("final stats: %s", stats.to_dict())
            try:
                log.info("journal summary: %s", engine.journal.summary())
            except Exception:
                pass

    if engine.risk.halted:
        # Exit non-zero so the supervisor can tell "the kill switch fired and the
        # state is deliberately halted" from "the process crashed and should come
        # back". The unit maps this code to a clean stop (SuccessExitStatus), so a
        # halted bot no longer restarts ten times, re-flattening an empty book,
        # before systemd marks the unit failed. The halt itself is persisted, so a
        # human has to reset it either way.
        log.error("halting: %s", engine.risk.halt_reason)
        return HALTED_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
