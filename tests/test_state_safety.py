"""Regression tests for the state-safety, risk-budget and secret-scanning fixes.

These are the tests whose absence let a bad startup wipe the position book, let a
malformed balance response empty it, and let an exit fail to free its capital.
Three of the four bugs below could destroy real money or real evidence, and every
one of them lived in a module with **zero** test coverage before.

On scratch directories: `tempfile.mkdtemp` is deliberately not used. It chmods the
new directory to 0700, and in a sandboxed process that sets a DACL the confined
token cannot then write through, so every test here would fail for an
environmental reason. A plain `mkdir` under the system temp directory works
everywhere including CI. Set ``ROOSTOO_TEST_SCRATCH`` to redirect the base if the
system temp area is itself read-only.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from typing import Iterator
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scan_secrets  # noqa: E402  (deliberately after the sys.path fix-up)
import check_encoding  # noqa: E402  (same: it lives in scripts/)

from roostoo.config import Config  # noqa: E402
from roostoo.engine import TradingEngine  # noqa: E402
from roostoo.models import (  # noqa: E402
    ExchangeInfo,
    OrderResult,
    Position,
    ShortPosition,
    Ticker,
    TradePair,
    WalletBalance,
)
from roostoo.risk import (  # noqa: E402
    ApprovedAction,
    PortfolioView,
    PositionBook,
    PositionSizer,
    RiskManager,
)
from roostoo.strategies.base import ENTER_LONG, EXIT_LONG, Signal  # noqa: E402
from roostoo.universe import RoostooDepthProvider  # noqa: E402

PAIR = "BTC/USD"
#: A credential-shaped value that is *not* on the allowlist, assembled from two
#: fragments so the literal never appears as one token in this file. That is not
#: paranoia for its own sake: `scripts/scan_secrets.py` scans the worktree, so a
#: single-line 60-character credential here would make every commit fail the
#: pre-commit hook -- and the tempting "fix" would be to allowlist this file,
#: which is exactly the hole that let a real key reach a public repository.
_FAKE_HEAD = "Qw3rTy9ZxCv2Bn4Mk6Lp8"
_FAKE_TAIL = "Rt1Yu3Io5Pa7Sd9Fg2Hj4Kl6Zx8Cv0Bn"
FAKE_SECRET = _FAKE_HEAD + _FAKE_TAIL


@contextlib.contextmanager
def scratch_dir() -> Iterator[Path]:
    """A writable scratch directory that works under a restricted sandbox."""
    base = Path(os.environ.get("ROOSTOO_TEST_SCRATCH") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"roostoo-test-{uuid.uuid4().hex[:12]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_config(directory: Path) -> Config:
    cfg = Config()
    cfg.mock = True
    cfg.api_key = "MOCK"
    cfg.secret_key = "MOCK"
    cfg.journal_dir = str(directory)
    cfg.log_dir = str(directory)
    return cfg


def trade_pair(pair: str = PAIR) -> TradePair:
    return TradePair(
        pair=pair,
        coin=pair.split("/")[0],
        unit="USD",
        can_trade=True,
        price_precision=2,
        amount_precision=6,
        min_order=10.0,
    )


def ticker(pair: str, mid: float = 100.0, volume: float = 1_000.0) -> Ticker:
    return Ticker(
        pair=pair,
        last=mid,
        max_bid=mid,
        min_ask=mid,
        change_24h=0.0,
        coin_volume=0.0,
        unit_volume=volume,
        server_time_ms=0,
    )


def write_book(path: Path, positions: dict) -> None:
    path.write_text(json.dumps({"positions": positions}), encoding="utf-8")


def book_fixture(quantity: float = 0.5, stop: float = 57_000.0) -> dict:
    return {
        "quantity": quantity,
        "avg_price": 60_000.0,
        "mark_price": 61_000.0,
        "is_short": False,
        "collateral": 0.0,
        "stop_price": stop,
        "take_profit_price": None,
        "peak_price": 62_000.0,
        "opened_ts_ms": 1_700_000_000_000,
    }


def write_state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "saved_ms": 1_700_000_000_000,
                "risk": {
                    "peak_nav": 123_456.0,
                    "day_id": 19_700,
                    "day_start_nav": 120_000.0,
                    "daily_halt": False,
                    "halted": True,
                    "halt_reason": "kill switch test",
                    "last_exit_bar": {"ETH/USD": 41},
                },
                "last_decision_bar": 56_666,
                "stats": {},
                "universe": [PAIR],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
class TestPositionBookCannotBeClobbered(unittest.TestCase):
    """``save()`` must never let an unloaded, empty book overwrite a real one."""

    def test_save_refuses_to_clobber_a_book_that_was_never_loaded(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            write_book(path, {PAIR: book_fixture()})
            before = path.read_bytes()

            # A fresh book, exactly as TradingEngine constructs one. `load()` is
            # never called, so `positions` is empty by default.
            PositionBook(path).save()

            self.assertEqual(path.read_bytes(), before, "an unloaded empty book overwrote a real one")

    def test_save_writes_normally_once_load_has_run(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            write_book(path, {PAIR: book_fixture()})
            book = PositionBook(path)
            self.assertTrue(book.load())
            self.assertIn(PAIR, book.positions)

            # Closing the last position is a legitimate reason to persist an
            # empty book: load() ran, so the emptiness is real information.
            book.positions.clear()
            book.save()
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["positions"], {})

    def test_save_works_when_no_book_exists_yet(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            PositionBook(path).save()
            self.assertTrue(path.is_file())

    def test_a_corrupt_book_is_not_clobbered_either(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            path.write_text("{ this is not json", encoding="utf-8")
            before = path.read_bytes()

            book = PositionBook(path)
            self.assertFalse(book.load())  # refuses to guess
            book.save()

            # Unreadable is not the same as empty: leave it for a human.
            self.assertEqual(path.read_bytes(), before)


# ---------------------------------------------------------------------------
class FlakyStartupClient:
    """Fails during bootstrap exactly the way a transient network error does."""

    def sync_time(self) -> int:
        raise RuntimeError("simulated transient network failure")

    def exchange_info(self) -> ExchangeInfo:
        raise AssertionError("must not be reached once sync_time has failed")

    def ticker(self, pair: str | None = None) -> dict:
        return {}

    def balance(self) -> dict:
        return {}

    def pending_count(self) -> tuple[int, dict]:
        return 0, {}

    def place_order(self, *args, **kwargs):  # pragma: no cover - no orders expected
        raise AssertionError("no orders expected")

    def query_orders(self, **kwargs) -> list:
        return []

    def cancel_order(self, *args, **kwargs) -> list:
        return []

    def short_positions(self) -> list:
        return []


class TestFailedStartupIsNonDestructive(unittest.TestCase):
    """The bug this pins: a startup blip used to erase the whole book."""

    def test_run_failure_leaves_the_stored_state_untouched(self) -> None:
        with scratch_dir() as d:
            book_path = d / "positions.json"
            state_path = d / "engine_state.json"
            write_book(book_path, {PAIR: book_fixture()})
            write_state(state_path)
            book_before = book_path.read_bytes()
            state_before = state_path.read_bytes()

            engine = TradingEngine(make_config(d), client=FlakyStartupClient())
            try:
                engine.run(max_cycles=1)
                self.fail("run() should have propagated the startup failure")
            except RuntimeError:
                pass
            finally:
                # Exactly what run_live.py does from its `finally:` block.
                engine.shutdown(flatten=False)

            self.assertEqual(book_path.read_bytes(), book_before, "the position book was wiped by a failed startup")
            self.assertEqual(state_path.read_bytes(), state_before, "the risk state was reset by a failed startup")

    def test_the_kill_switch_survives_a_failed_startup(self) -> None:
        """`halted` must not be cleared by a blip: that would un-halt a halted bot."""
        with scratch_dir() as d:
            write_book(d / "positions.json", {PAIR: book_fixture()})
            write_state(d / "engine_state.json")

            engine = TradingEngine(make_config(d), client=FlakyStartupClient())
            try:
                engine.run(max_cycles=1)
            except RuntimeError:
                pass
            finally:
                engine.shutdown(flatten=False)

            restored = json.loads((d / "engine_state.json").read_text(encoding="utf-8"))
            self.assertTrue(restored["risk"]["halted"])
            self.assertEqual(restored["risk"]["peak_nav"], 123_456.0)
            self.assertEqual(restored["risk"]["last_exit_bar"], {"ETH/USD": 41})

    def test_persist_is_gated_before_bootstrap(self) -> None:
        with scratch_dir() as d:
            engine = TradingEngine(make_config(d), client=FlakyStartupClient())
            engine._persist()
            self.assertFalse((d / "positions.json").exists())
            self.assertFalse((d / "engine_state.json").exists())


# ---------------------------------------------------------------------------
class NoUsdBalanceClient:
    """A partial balance snapshot: assets are reported, the quote currency is not."""

    def sync_time(self) -> int:
        return 0

    def exchange_info(self) -> ExchangeInfo:
        return ExchangeInfo(is_running=True, initial_wallet={"USD": 100_000.0}, pairs={PAIR: trade_pair()})

    def ticker(self, pair: str | None = None) -> dict:
        return {PAIR: ticker(PAIR)}

    def balance(self) -> dict:
        # No "USD" row at all. `_cash_usd` would return 0.0, NAV would collapse to
        # the position marks, and the drawdown check would trip the permanent
        # kill switch; reconciliation would also read every missing row as "the
        # exchange holds nothing" and delete the book.
        return {"BTC": WalletBalance(asset="BTC", free=0.5, locked=0.0)}

    def pending_count(self) -> tuple[int, dict]:
        return 0, {}

    def place_order(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("no orders expected")

    def query_orders(self, **kwargs) -> list:
        return []

    def cancel_order(self, *args, **kwargs) -> list:
        return []

    def short_positions(self) -> list:
        return []


class TestIncompleteBalanceSnapshotIsNotActedOn(unittest.TestCase):
    def _engine(self, directory: Path) -> TradingEngine:
        engine = TradingEngine(make_config(directory), client=NoUsdBalanceClient())
        self.addCleanup(engine.journal.close)  # do not leak the journal handles
        engine.exchange_pairs = {PAIR: trade_pair()}
        engine.book.positions[PAIR] = Position(
            pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=61_000.0, stop_price=57_000.0
        )
        return engine

    def test_cycle_is_skipped_and_the_book_is_preserved(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d)
            engine.step()

            self.assertIn(PAIR, engine.book.positions, "a partial balance snapshot deleted the position")
            self.assertEqual(engine.book.positions[PAIR].stop_price, 57_000.0)
            self.assertFalse(engine.risk.halted, "a partial snapshot tripped the permanent kill switch")

    def test_the_skip_is_journalled(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d)
            engine.step()

            from roostoo.journal import read_events

            events = []
            for path in sorted(Path(d).glob("decisions-*.jsonl")):
                events += read_events(path)
            self.assertTrue(
                any(e.get("kind") == "error" and "USD" in str(e.get("message", "")) for e in events),
                f"expected a journalled reason for the skipped cycle, got {[e.get('kind') for e in events]}",
            )


# ---------------------------------------------------------------------------
class TestGrossBudgetIsFreedBySameBatchExits(unittest.TestCase):
    """Rule 10 freed the slot but Rule 9 kept rejecting the entry."""

    HELD = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD")
    NEW = "ADA/USD"

    def _view(self) -> PortfolioView:
        # 4 x 15% == exactly the 60% gross cap.
        positions = {
            p: Position(pair=p, quantity=150.0, avg_price=100.0, mark_price=100.0) for p in self.HELD
        }
        return PortfolioView(nav=100_000.0, cash_usd=40_000.0, positions=positions)

    def test_an_exit_earlier_in_the_batch_frees_capital_for_a_later_entry(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        tickers = {p: ticker(p) for p in self.HELD + (self.NEW,)}

        decision = risk.evaluate(
            [
                Signal("BTC/USD", EXIT_LONG, reason="stop hit"),
                Signal(self.NEW, ENTER_LONG, meta={"atr": 1.0}),
            ],
            view=self._view(),
            tickers=tickers,
            now_ms=0,
            bar_idx=1,
        )
        approved = {a.pair: a.action for a in decision.approved}
        self.assertEqual(approved.get("BTC/USD"), EXIT_LONG)
        self.assertEqual(
            approved.get(self.NEW),
            ENTER_LONG,
            f"the exit freed a slot and 15k of gross, but the entry was still rejected: {decision.rejected}",
        )

    def test_a_batch_cannot_overshoot_the_gross_cap(self) -> None:
        """The fix must free budget *without* letting a batch exceed Rule 9."""
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        pairs = ["A/USD", "B/USD", "C/USD", "D/USD", "E/USD"]
        tickers = {p: ticker(p) for p in pairs}
        view = PortfolioView(nav=100_000.0, cash_usd=100_000.0, positions={})

        decision = risk.evaluate(
            [Signal(p, ENTER_LONG, meta={"atr": 1.0}) for p in pairs],
            view=view,
            tickers=tickers,
            now_ms=0,
            bar_idx=1,
        )
        total = sum(a.notional for a in decision.approved if a.is_entry)
        self.assertLessEqual(total, 100_000.0 * cfg.max_gross_exposure + 1e-6)
        self.assertLessEqual(len(decision.approved), cfg.max_open_positions)


# ---------------------------------------------------------------------------
class TestInFlightOrdersReserveCapital(unittest.TestCase):
    """A pending order is not a position yet, but its capital must be committed.

    Without this, two consecutive bars each see a flat account and each size a
    full position for the same pair, ending at twice the per-pair cap.
    """

    def _evaluate(self, committed_pairs: set[str]):
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        tickers = {PAIR: ticker(PAIR)}
        view = PortfolioView(nav=100_000.0, cash_usd=100_000.0, positions={})
        return risk.evaluate(
            [Signal(PAIR, ENTER_LONG, meta={"atr": 1.0})],
            view=view,
            tickers=tickers,
            now_ms=0,
            bar_idx=1,
            committed_pairs=committed_pairs,
            committed_notional=15_000.0 if committed_pairs else 0.0,
        )

    def test_an_entry_is_approved_when_nothing_is_in_flight(self) -> None:
        decision = self._evaluate(set())
        self.assertEqual([a.pair for a in decision.approved], [PAIR])

    def test_an_entry_is_refused_for_a_pair_already_in_flight(self) -> None:
        decision = self._evaluate({PAIR})
        self.assertEqual(decision.approved, [], "a second full-size entry was approved for an in-flight pair")
        self.assertTrue(
            any("already holding" in reason for _, reason in decision.rejected),
            f"expected a reservation rejection, got {decision.rejected}",
        )


# ---------------------------------------------------------------------------
class RecordingTransport:
    """Captures the exact request the depth provider sends."""

    def __init__(self, payload: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.payload = payload if payload is not None else {"Bids": [[99.0, 2.0]], "Asks": [[101.0, 2.0]]}

    def send(self, method, url, body, headers, timeout):
        self.calls.append((method, url, body, dict(headers), timeout))
        return 200, json.dumps(self.payload)


class SigningClientStub:
    base_url = "https://example.invalid"

    def __init__(self) -> None:
        self.signed: list[str | None] = []

    def timestamp_ms(self) -> str:
        return "1580774512000"

    def sign_headers(self, params, canonical=None):
        self.signed.append(canonical)
        return {"RST-API-KEY": "KEY", "MSG-SIGNATURE": "SIGNED"}


class TestDepthProviderSignsItsRequests(unittest.TestCase):
    """It used to probe for a private `_sign_headers` that never existed."""

    def test_signature_covers_the_exact_query_string_that_is_sent(self) -> None:
        client = SigningClientStub()
        transport = RecordingTransport()
        provider = RoostooDepthProvider(client, "/v3/depth", transport=transport, timeout=5.0)

        snapshot = provider.snapshot(PAIR, 0.005)

        self.assertIsNotNone(snapshot)
        self.assertEqual(len(transport.calls), 1)
        _, url, _, headers, _ = transport.calls[0]
        self.assertEqual(headers.get("MSG-SIGNATURE"), "SIGNED", "the request was sent unsigned")
        self.assertEqual(headers.get("RST-API-KEY"), "KEY")
        # The signed string must be byte-identical to the query actually sent,
        # otherwise the server rebuilds a different canonical string and rejects it.
        self.assertEqual(client.signed[0], url.split("?", 1)[1])

    def test_a_client_without_signing_still_works(self) -> None:
        class UnsignedClient:
            base_url = "https://example.invalid"

        transport = RecordingTransport()
        provider = RoostooDepthProvider(UnsignedClient(), "/v3/depth", transport=transport, timeout=5.0)
        self.assertIsNotNone(provider.snapshot(PAIR, 0.005))
        self.assertNotIn("MSG-SIGNATURE", transport.calls[0][3])


# ---------------------------------------------------------------------------
class TestTextEncodingGuard(unittest.TestCase):
    """A re-encoded file is still valid UTF-8, so nothing else catches this.

    `Get-Content README.md | Set-Content README.md -Encoding utf8` in Windows
    PowerShell decodes UTF-8 as the machine's ANSI codepage and re-encodes the
    result: every em dash became a CJK lookalike (U+9225, among others) and a BOM
    appeared at the top of the file. Every test passed, `compileall` passed, and
    the damage was only visible as mojibake on GitHub. `scripts/check_encoding.py`
    is the guard.
    """

    def test_the_real_repository_is_clean(self) -> None:
        files = check_encoding.tracked_text_files([])
        self.assertGreater(len(files), 10, "the guard found almost no files to check")
        damaged = {str(p): check_encoding.check_file(p) for p in files if p.is_file()}
        damaged = {name: problems for name, problems in damaged.items() if problems}
        self.assertEqual(damaged, {}, f"these files are not clean UTF-8: {damaged}")

    def test_a_bom_is_caught(self) -> None:
        with scratch_dir() as d:
            path = d / "bom.md"
            path.write_bytes(b"\xef\xbb\xbf# Title\n")
            problems = check_encoding.check_file(path)
            self.assertTrue(any("BOM" in p for p in problems), problems)

    def test_mojibake_is_caught(self) -> None:
        with scratch_dir() as d:
            path = d / "mojibake.md"
            # Exactly what a cp936 round trip did to "Rules 1-3 -- the entry logic":
            # U+9225 is the em dash artefact, U+951B the fullwidth-colon artefact.
            path.write_text("Rules 1\u92252\u2014the entry \u951b\u9286 logic\n", encoding="utf-8")
            problems = check_encoding.check_file(path)
            self.assertTrue(problems, "a cp936 artefact was accepted")

    def test_the_sigma_artefact_is_caught(self) -> None:
        """A cp936 misread of a 2-byte UTF-8 sequence lands inside the CJK range."""
        with scratch_dir() as d:
            path = d / "sigma.md"
            path.write_text("Z = \u87fd\n", encoding="utf-8")
            self.assertTrue(check_encoding.check_file(path))

    def test_a_private_use_character_is_caught(self) -> None:
        with scratch_dir() as d:
            path = d / "pua.md"
            path.write_text("value \ue0a2 table\n", encoding="utf-8")
            self.assertTrue(check_encoding.check_file(path))

    def test_legitimate_content_is_not_flagged(self) -> None:
        """The guard must not police the languages the docs legitimately use."""
        with scratch_dir() as d:
            path = d / "good.md"
            path.write_text(
                "# Title \u2014 with an em dash\n"
                "Z \u2264 \u22122\u03c3 and \u00b10.5% and \u0394Z > 0\n"
                "## 8. \u4e2d\u6587\u5feb\u901f\u5f00\u59cb\n"
                "\u65e0\u5bc6\u94a5\u5148\u8dd1\u901a\u6574\u6761\u94fe\u8def\uff08\u5185\u7f6e\u6a21\u62df\u4ea4\u6613\u6240\uff09\n"
                "| exit | n | \u00b1\u2014\u2264\u2265 |\n",
                encoding="utf-8",
            )
            self.assertEqual(check_encoding.check_file(path), [])

    def test_ci_runs_the_guard(self) -> None:
        workflow = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/check_encoding.py", workflow)

    def test_the_guard_exempts_only_itself(self) -> None:
        """It quotes the artefacts it detects, so it must skip its own file -- and
        nothing else. Every entry here is a place a real bug could hide."""
        self.assertEqual(check_encoding.SELF_EXEMPT, {"scripts/check_encoding.py"})
        self.assertTrue(check_encoding._skip(Path("scripts/check_encoding.py")))
        self.assertFalse(check_encoding._skip(Path("README.md")))
        self.assertFalse(check_encoding._skip(Path("scripts/scan_secrets.py")))


# ---------------------------------------------------------------------------
class TestStateScopeIsolation(unittest.TestCase):
    """mock and live share the default journal directory.

    A state file therefore has to record which runtime wrote it. Without that, a
    simulated book -- positions, stop levels, the drawdown high-water mark, a
    simulated halt -- is indistinguishable from a real one and is adopted by live.
    A file written before the stamp existed is still accepted, because refusing it
    on an upgrade would strand real positions without their stops; only an
    explicit contradiction is refused.
    """

    @staticmethod
    def _write_scoped(path: Path, scope: dict[str, str]) -> None:
        write_book(path, {PAIR: book_fixture()})
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scope"] = scope
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_scope_names_the_mode_and_the_venue(self) -> None:
        with scratch_dir() as d:
            cfg = make_config(d)  # make_config sets mock = True
            self.assertEqual(cfg.state_scope["mode"], "mock")
            cfg.mock = False
            cfg.base_url = "https://api.example.com"
            self.assertEqual(
                cfg.state_scope, {"mode": "live", "venue": "https://api.example.com"}
            )

    def test_a_mock_book_is_refused_by_a_live_process(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            self._write_scoped(path, {"mode": "mock", "venue": "https://mock-api.roostoo.com"})
            cfg = make_config(d)
            cfg.mock = False
            cfg.base_url = "https://mock-api.roostoo.com"
            book = PositionBook(path, cfg=cfg)
            self.assertFalse(book.load(), "live adopted a simulated book")
            self.assertEqual(book.positions, {})

    def test_a_second_live_venue_is_refused_too(self) -> None:
        """The test venue and the competition venue are both 'live', so mode alone
        is not enough to identify whose stops these are."""
        with scratch_dir() as d:
            path = d / "positions.json"
            self._write_scoped(path, {"mode": "live", "venue": "https://mock-api.roostoo.com"})
            cfg = make_config(d)
            cfg.mock = False
            cfg.base_url = "https://competition.example.com"
            book = PositionBook(path, cfg=cfg)
            self.assertFalse(book.load())
            self.assertEqual(book.positions, {})

    def test_the_matching_scope_loads(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            cfg = make_config(d)
            cfg.mock = False
            self._write_scoped(path, cfg.state_scope)
            book = PositionBook(path, cfg=cfg)
            self.assertTrue(book.load())
            self.assertIn(PAIR, book.positions)

    def test_an_unstamped_book_still_loads(self) -> None:
        with scratch_dir() as d:
            path = d / "positions.json"
            write_book(path, {PAIR: book_fixture()})
            cfg = make_config(d)
            book = PositionBook(path, cfg=cfg)
            self.assertTrue(book.load())
            self.assertIn(PAIR, book.positions)

    def test_a_refused_book_can_still_be_replaced(self) -> None:
        """Refusing to load is not refusing to save. If the stale file could not be
        overwritten, the wrong-venue state would sit there forever."""
        with scratch_dir() as d:
            path = d / "positions.json"
            self._write_scoped(path, {"mode": "mock", "venue": "elsewhere"})
            cfg = make_config(d)
            cfg.mock = False
            book = PositionBook(path, cfg=cfg)
            self.assertFalse(book.load())
            book.save()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("scope"), cfg.state_scope)
            self.assertEqual(payload["positions"], {})

    def test_a_book_without_a_config_is_unaffected(self) -> None:
        """Tests and tools construct PositionBook(path) with no cfg; the check must
        not fire for them."""
        with scratch_dir() as d:
            path = d / "positions.json"
            self._write_scoped(path, {"mode": "mock", "venue": "elsewhere"})
            book = PositionBook(path)
            self.assertTrue(book.load())

    def test_the_engine_scope_gate(self) -> None:
        with scratch_dir() as d:
            cfg = make_config(d)  # mock
            engine = mock.MagicMock(cfg=cfg)
            self.assertTrue(TradingEngine._scope_matches(engine, cfg.state_scope))
            self.assertFalse(
                TradingEngine._scope_matches(
                    engine, {"mode": "live", "venue": cfg.base_url}
                )
            )


# ---------------------------------------------------------------------------
class TestLimitEntries(unittest.TestCase):
    """Maker entries: post a resting bid, and cancel it if it goes stale.

    `docs/FINDINGS.md` 6b.1 measures this as worth +0.047% per trade -- real, and
    about a seventh of the loss. These tests pin the mechanics, not the benefit;
    the benefit has to be re-measured in the live journal, because the simulator
    does not model a resting bid being filled by an intrabar low.
    """

    def _engine(self, directory: Path, **cfg_overrides: object) -> tuple[TradingEngine, _StatefulClient]:
        client = _StatefulClient()
        cfg = make_config(directory)
        for key, value in cfg_overrides.items():
            setattr(cfg, key, value)
        engine = TradingEngine(cfg, client=client)
        self.addCleanup(engine.journal.close)
        engine.exchange_pairs = {PAIR: trade_pair()}
        engine.tickers = {PAIR: ticker(PAIR, 100.0)}
        return engine, client

    def _action(self) -> ApprovedAction:
        return ApprovedAction(pair=PAIR, action=ENTER_LONG, quantity=1.0, notional=100.0, reason="test")

    def test_the_limit_price_is_the_mid_rounded_to_the_tick(self) -> None:
        with scratch_dir() as d:
            engine, _ = self._engine(d, limit_entries=True)
            pair = engine.exchange_pairs[PAIR]
            engine.tickers[PAIR] = ticker(PAIR, 100.456)
            self.assertEqual(engine._limit_price(PAIR, pair, engine.tickers[PAIR]), 100.45)

    def test_a_positive_offset_bids_below_the_mid(self) -> None:
        with scratch_dir() as d:
            engine, _ = self._engine(d, limit_entries=True, limit_entry_offset_bps=10.0)
            pair = engine.exchange_pairs[PAIR]
            engine.tickers[PAIR] = ticker(PAIR, 100.0)
            # 10bp below 100.00 is 99.90
            self.assertEqual(engine._limit_price(PAIR, pair, engine.tickers[PAIR]), 99.90)

    def test_entries_are_posted_as_limit_orders_when_enabled(self) -> None:
        with scratch_dir() as d:
            engine, client = self._engine(d, limit_entries=True)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            self.assertEqual(len(client.orders_placed), 1)
            _pair, _side, _qty, order_type, price = client.orders_placed[0]
            self.assertEqual(order_type, "LIMIT")
            self.assertAlmostEqual(price, 100.0, places=2)
            self.assertIn(PAIR, engine._resting)
            self.assertEqual(engine.stats.orders_posted, 1)

    def test_entries_stay_market_orders_by_default(self) -> None:
        with scratch_dir() as d:
            engine, client = self._engine(d)  # limit_entries defaults to False
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            self.assertEqual(client.orders_placed[0][3], "MARKET")
            self.assertEqual(engine._resting, {})

    def test_a_stale_bid_is_cancelled_and_forgotten(self) -> None:
        with scratch_dir() as d:
            engine, client = self._engine(d, limit_entries=True, limit_entry_timeout_bars=1)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            self.assertIn(PAIR, engine._resting)

            # Same bar: still fresh, must survive.
            engine._expire_resting_orders(1_800_000_000_000, 5)
            self.assertIn(PAIR, engine._resting)

            # One bar later: stale, so it is cancelled.
            engine._expire_resting_orders(1_800_000_060_000, 6)
            self.assertNotIn(PAIR, engine._resting)
            self.assertEqual(client.cancelled, [PAIR])
            self.assertEqual(engine.stats.orders_cancelled, 1)

    def test_a_cancel_that_fails_keeps_the_order_tracked(self) -> None:
        """Assuming a failed cancel happened would leak the reservation forever."""
        with scratch_dir() as d:
            engine, client = self._engine(d, limit_entries=True, limit_entry_timeout_bars=1)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            client.cancel_raises = True
            engine._expire_resting_orders(1_800_000_060_000, 6)
            self.assertIn(PAIR, engine._resting, "a failed cancel must be retried, not assumed")
            self.assertEqual(engine.stats.orders_cancelled, 0)

    def test_only_one_bid_is_posted_per_pair(self) -> None:
        with scratch_dir() as d:
            engine, client = self._engine(d, limit_entries=True)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            self.assertEqual(len(client.orders_placed), 1, "a second bid was stacked on the first")

    def test_sizing_uses_the_limit_price_not_the_mid(self) -> None:
        """A fill below the mid at mid-based sizing would exceed the approved cap."""
        with scratch_dir() as d:
            engine, client = self._engine(d, limit_entries=True, limit_entry_offset_bps=100.0)
            engine._enter_long(self._action(), 1_800_000_000_000, 5)
            _pair, _side, qty, _type, price = client.orders_placed[0]
            self.assertAlmostEqual(price, 99.0, places=2)  # 100bp below 100
            # quantity * limit price must not exceed the approved 100.0 notional
            self.assertLessEqual(float(qty) * price, 100.0 + 1e-9)



# ---------------------------------------------------------------------------
class TestEdgeAnalysisTool(unittest.TestCase):
    """The analysis tool decides whether a change is worth making, so it is tested.

    Its most dangerous property is the fill assumption: measuring from the signal
    bar's own close instead of the next bar's was worth 0.13 percentage points per
    trade -- four times the edge being measured -- and it flipped the conclusion.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import edge_analysis

        self.tool = edge_analysis

    def test_the_fill_happens_one_bar_after_the_signal(self) -> None:
        """The measured return must start at the FILL bar, not the signal bar."""
        n = self.tool.HORIZON_BARS + 4
        # Signal on bar 1 (close 100). The next bar closes at 200 and nothing moves
        # after that, so a correctly-filled trade returns exactly 0 while a trade
        # filled at the signal bar's own close would report +100%.
        closes = [10.0, 100.0] + [200.0] * (n - 2)
        z = [0.0, -2.0] + [-1.5] * (n - 2)
        self.assertAlmostEqual(self.tool.forward_return(1, closes, z), 0.0, places=12)

    def test_the_target_exit_is_used_when_it_comes_first(self) -> None:
        n = self.tool.HORIZON_BARS + 4
        closes = [10.0, 100.0, 100.0, 150.0] + [120.0] * (n - 4)
        z = [0.0, -2.0, -2.0, -0.1] + [0.0] * (n - 4)
        # Filled at 100 on bar 2, exits at the -0.25 target on bar 3 at 150.
        self.assertAlmostEqual(self.tool.forward_return(1, closes, z), 0.5, places=12)

    def test_match_span_keeps_the_wall_clock_window(self) -> None:
        """48 bars of 30m is 24h, so 4h bars need a 6-bar window, not 48."""
        self.assertEqual(self.tool.span_window("30m"), 48)
        self.assertEqual(self.tool.span_window("4h"), 6)

    def test_an_unknown_interval_is_rejected_loudly(self) -> None:
        with self.assertRaises(SystemExit):
            self.tool.span_window("7m")

    def test_the_cost_model_matches_the_rulebook(self) -> None:
        self.assertAlmostEqual(self.tool.TAKER, 0.001)
        self.assertAlmostEqual(self.tool.MAKER, 0.0005)

    # -- the passive-fill model ------------------------------------------
    def _series(self, highs, lows, closes, name: str = "BTC-USD_30m.csv") -> None:
        self.tool._SERIES[name] = (highs, lows, closes)

    def test_a_touch_fills_and_a_miss_does_not(self) -> None:
        """The limit is the signal bar's close; bar i+1 decides whether it fills."""
        n = self.tool.HORIZON_BARS + 4
        closes = [100.0] * n
        z = [-2.0] + [-1.0] * (n - 1)

        # The fill bar dips to exactly the limit -> a 'touch' fill, no 'through' fill.
        lows = [100.0] * n
        lows[1] = 100.0
        highs = [101.0] * n
        self._series(highs, lows, closes)
        _ret, touched = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 0.0, "touch")
        _ret2, through = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 0.0, "through")
        self.assertTrue(touched, "touching the limit should fill under the 'touch' model")
        self.assertFalse(through, "touching but not penetrating must not fill 'through'")

    def test_a_bar_that_never_reaches_the_limit_does_not_fill(self) -> None:
        n = self.tool.HORIZON_BARS + 4
        closes = [100.0] * n
        z = [-2.0] + [-1.0] * (n - 1)
        highs = [105.0] * n
        lows = [101.0] * n  # stays above the 100 limit
        self._series(highs, lows, closes)
        _ret, filled = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 0.0, "touch")
        self.assertFalse(filled)

    def test_the_offset_lowers_the_resting_bid(self) -> None:
        """A 10bp offset bids 0.1% below the signal close, so it fills less often."""
        n = self.tool.HORIZON_BARS + 4
        closes = [100.0] * n
        z = [-2.0] + [-1.0] * (n - 1)
        highs = [101.0] * n
        lows = [100.0] * n
        self._series(highs, lows, closes)
        _r, at_market = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 0.0, "touch")
        _r2, ten_bp = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 10.0, "touch")
        self.assertTrue(at_market)
        self.assertFalse(ten_bp, "a bid 10bp lower should not fill on a bar that only reached 100")

    def test_a_passive_fill_returns_the_target_move_from_the_limit(self) -> None:
        """Entry is the limit (passive), exit is the same target as the market case."""
        n = self.tool.HORIZON_BARS + 4
        closes = [100.0] * n
        z = [-2.0] + [-1.0] * (n - 1)
        closes[3] = 110.0        # reaches the target on the exit search
        z[3] = -0.1
        highs = [101.0] * n
        lows = [100.0] * n
        self._series(highs, lows, closes)
        ret, filled = self.tool.maker_entry("BTC-USD_30m.csv", 0, closes, z, 0.0, "touch")
        self.assertTrue(filled)
        self.assertAlmostEqual(ret, 0.10, places=9)

    def test_an_unknown_fill_model_is_rejected(self) -> None:
        n = self.tool.HORIZON_BARS + 4
        self._series([1.0] * n, [1.0] * n, [1.0] * n)
        with self.assertRaises(ValueError):
            self.tool.maker_entry("BTC-USD_30m.csv", 0, [1.0] * n, [-2.0] * n, 0.0, "guess")


# ---------------------------------------------------------------------------
class TestSecretScanner(unittest.TestCase):
    """The old guard read filenames; this one reads bytes."""

    def test_a_hardcoded_credential_is_caught(self) -> None:
        findings = scan_secrets.scan_text(f'API_KEY = "{FAKE_SECRET}"', "roostoo/whatever.py")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "credential-assignment")

    def test_the_finding_never_echoes_the_secret(self) -> None:
        """A redaction that truncates first would print the head of the key."""
        findings = scan_secrets.scan_text(f'API_KEY = "{FAKE_SECRET}"', "x.py")
        self.assertNotIn(FAKE_SECRET, findings[0].excerpt)
        self.assertNotIn(FAKE_SECRET[:16], findings[0].excerpt)
        self.assertIn("<redacted>", findings[0].excerpt)

    def test_the_published_docs_vector_is_allowlisted(self) -> None:
        doc_secret = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"
        self.assertIn(doc_secret, scan_secrets.ALLOWED_VALUES)
        self.assertEqual(scan_secrets.scan_text(f'SECRET_KEY = "{doc_secret}"', "tests/test_client.py"), [])

    def test_placeholders_and_templates_are_not_flagged(self) -> None:
        for line in (
            'API_KEY = "your_api_key_here_placeholder"',
            "ROOSTOO_SECRET_KEY=",
            "SECRET_KEY = ''",
            'api_key = "${ROOSTOO_API_KEY}"',
        ):
            with self.subTest(line=line):
                self.assertEqual(scan_secrets.scan_text(line, "x.py"), [])

    #: Assembled at runtime so this file does not itself contain a literal PEM
    #: header. A literal one is a permanent false positive for the worktree scan
    #: *and* the history scan, and the tempting cure for that is to allowlist
    #: this file -- which is precisely how a real key would get through.
    _PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"

    def test_private_key_blocks_are_caught(self) -> None:
        findings = scan_secrets.scan_text(self._PEM_HEADER, "id_rsa")
        self.assertEqual([f.kind for f in findings], ["private-key-block"])

    def test_the_scanners_own_pattern_list_is_not_a_finding(self) -> None:
        """It contains the word `secret` and a PEM header; it must skip itself."""
        self.assertTrue(scan_secrets._skip("scripts/scan_secrets.py"))
        self.assertEqual(scan_secrets.scan_worktree(), [])


# ---------------------------------------------------------------------------
# State safety, round two. Every case here is a bug that shipped: a transient
# error that deleted live positions, an absent balance row read as "flat", a
# history row with no status read as a fill, and a malformed state file that
# stopped the bot from starting at all.
# ---------------------------------------------------------------------------


class _StatefulClient:
    """Minimal client whose failure modes are set per test."""

    def __init__(self, short_rows: list | None = None, order_rows: list | None = None) -> None:
        self._short_rows = short_rows if short_rows is not None else []
        self._order_rows = order_rows if order_rows is not None else []
        self.short_positions_raises = False
        self.cancel_raises = False
        #: (pair, side, qty, order_type, price) for every order attempted.
        self.orders_placed: list[tuple] = []
        #: pairs passed to cancel_order.
        self.cancelled: list[str] = []

    def sync_time(self) -> int:
        return 0

    def exchange_info(self) -> ExchangeInfo:
        return ExchangeInfo(pairs={PAIR: trade_pair()}, is_running=True, initial_wallet=100_000.0)

    def ticker(self, pair: str | None = None) -> dict:
        return {PAIR: ticker(PAIR)}

    def balance(self) -> dict:
        return {"USD": WalletBalance(asset="USD", free=100_000.0, locked=0.0)}

    def pending_count(self) -> tuple[int, dict]:
        return 0, {}

    def place_order(self, pair, side, qty, order_type="MARKET", price=None):
        self.orders_placed.append((pair, side, qty, order_type, price))
        # A limit order that does not cross rests, exactly as the real client and
        # the simulator both report it. Modelling it as an instant fill would make
        # the maker path look like free money.
        resting = order_type.upper() == "LIMIT" and price is not None
        return OrderResult(
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=float(qty),
            price=float(price or 0.0),
            status="PENDING" if resting else "FILLED",
            filled_quantity=0.0 if resting else float(qty),
            avg_fill_price=0.0 if resting else 60_000.0,
        )

    def query_orders(self, **kwargs) -> list:
        return list(self._order_rows)

    def cancel_order(self, *args, **kwargs) -> list:
        if self.cancel_raises:
            raise RuntimeError("simulated cancel failure")
        pair = kwargs.get("pair")
        if pair:
            self.cancelled.append(pair)
        return []

    def short_positions(self) -> list:
        if self.short_positions_raises:
            raise RuntimeError("simulated transient failure")
        return list(self._short_rows)


def make_engine(directory: Path, client) -> TradingEngine:
    engine = TradingEngine(make_config(directory), client=client)
    engine.exchange_pairs = {PAIR: trade_pair()}
    return engine


class TestUnknownOrderReconciliationDoesNotInventFills(unittest.TestCase):
    """`_reconcile_unknown_order` had zero test coverage and matched on recency."""

    def _action(self) -> ApprovedAction:
        return ApprovedAction(pair=PAIR, action=ENTER_LONG, quantity=0.5, notional=50.0, reason="test")

    def _unknown(self) -> OrderResult:
        return OrderResult(
            pair=PAIR, side="BUY", order_type="MARKET", quantity=0.5, price=0.0, status="UNKNOWN", err_msg="timeout"
        )

    def _row(self, **overrides) -> dict:
        row = {
            "CreateTimestamp": int(time.time() * 1000) - 1_000,
            "Side": "BUY",
            "Status": "FILLED",
            "Quantity": 0.5,
            "FilledQuantity": 0.5,
            "FilledAverPrice": 60_100.0,
            "OrderID": 7,
        }
        row.update(overrides)
        return row

    def test_a_genuine_fill_is_still_recognised(self) -> None:
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient(order_rows=[self._row()]))
            self.addCleanup(engine.journal.close)
            engine._record_result(self._unknown(), self._action(), int(time.time() * 1000), 1)
            position = engine.book.get(PAIR)
            self.assertIsNotNone(position, "a real fill must still be booked")
            self.assertAlmostEqual(position.quantity, 0.5)
            self.assertAlmostEqual(position.avg_price, 60_100.0)

    def test_a_row_with_no_status_is_not_a_fill(self) -> None:
        """The row that used to become a 7.0-unit position at price zero."""
        row = self._row()
        del row["Status"]
        row["Quantity"] = 7.0
        row["FilledQuantity"] = 0.0
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient(order_rows=[row]))
            self.addCleanup(engine.journal.close)
            self.assertIsNone(engine._reconcile_unknown_order(self._action(), self._unknown(), int(time.time() * 1000)))
            engine._record_result(self._unknown(), self._action(), int(time.time() * 1000), 1)
            self.assertEqual(engine.book.positions, {}, "a row with no Status produced a phantom position")

    def test_a_cancelled_row_of_another_size_is_not_ours(self) -> None:
        row = self._row(Status="CANCELED", Quantity=999.0, FilledQuantity=0.0)
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient(order_rows=[row]))
            self.addCleanup(engine.journal.close)
            self.assertIsNone(engine._reconcile_unknown_order(self._action(), self._unknown(), int(time.time() * 1000)))

    def test_a_filled_status_with_no_quantity_is_not_booked(self) -> None:
        row = self._row(FilledQuantity=0.0)
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient(order_rows=[row]))
            self.addCleanup(engine.journal.close)
            engine._record_result(self._unknown(), self._action(), int(time.time() * 1000), 1)
            self.assertEqual(engine.book.positions, {}, "FILLED with nothing filled was booked anyway")


class TestShortPositionsSurviveATransientError(unittest.TestCase):
    def test_a_failed_query_does_not_delete_the_books_shorts(self) -> None:
        with scratch_dir() as d:
            client = _StatefulClient()
            client.short_positions_raises = True
            engine = make_engine(d, client)
            self.addCleanup(engine.journal.close)
            engine.book.positions[PAIR] = Position(
                pair=PAIR, is_short=True, quantity=0.5, avg_price=100.0, collateral=50.0, stop_price=110.0
            )
            self.assertIsNone(engine._safe_short_positions(), "an unanswered query must not read as 'no shorts'")
            engine._reconcile_positions({"USD": WalletBalance(asset="USD", free=90_000.0, locked=0.0)}, None)
            self.assertIn(PAIR, engine.book.positions, "a transient error deleted a live short")

    def test_a_real_short_at_the_venue_is_adopted(self) -> None:
        """An unbooked short used to be invisible to NAV and every cap."""
        row = ShortPosition(
            position_id=1,
            pair=PAIR,
            entry_price=100.0,
            quantity=0.5,
            collateral=50.0,
            current_price=100.0,
            unrealized_pnl=0.0,
            unrealized_pct=0.0,
            position_value=50.0,
            created_ts_ms=1,
        )
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient(short_rows=[row]))
            self.addCleanup(engine.journal.close)
            engine._reconcile_positions({"USD": WalletBalance(asset="USD", free=90_000.0, locked=0.0)}, [row])
            position = engine.book.get(PAIR)
            self.assertIsNotNone(position, "a short the venue reports must be booked")
            self.assertTrue(position.is_short)
            self.assertAlmostEqual(position.quantity, 0.5)

    def test_a_confirmed_empty_answer_still_closes_the_book_short(self) -> None:
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            engine.book.positions[PAIR] = Position(pair=PAIR, is_short=True, quantity=0.5, avg_price=100.0)
            engine._reconcile_positions({"USD": WalletBalance(asset="USD", free=90_000.0, locked=0.0)}, [])
            self.assertNotIn(PAIR, engine.book.positions)


class TestAbsentBalanceRowIsNotAClose(unittest.TestCase):
    def test_a_missing_coin_row_keeps_the_position(self) -> None:
        """USD present and the coin row gone: the old guard waved this through."""
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            engine.book.positions[PAIR] = Position(
                pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=61_000.0, stop_price=57_000.0
            )
            engine._reconcile_positions({"USD": WalletBalance(asset="USD", free=90_000.0, locked=0.0)}, [])
            self.assertIn(PAIR, engine.book.positions, "an absent row deleted the whole book")
            self.assertEqual(engine.book.positions[PAIR].stop_price, 57_000.0)

    def test_an_explicit_zero_row_still_closes_it(self) -> None:
        with scratch_dir() as d:
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            engine.book.positions[PAIR] = Position(pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=61_000.0)
            engine._reconcile_positions(
                {
                    "USD": WalletBalance(asset="USD", free=90_000.0, locked=0.0),
                    "BTC": WalletBalance(asset="BTC", free=0.0, locked=0.0),
                },
                [],
            )
            self.assertNotIn(PAIR, engine.book.positions)


class TestAStopCannotSilentlyVanish(unittest.TestCase):
    def test_a_non_finite_atr_refuses_the_entry(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        sizer = PositionSizer(cfg)
        sizing = sizer.size(nav=100_000.0, price=101.0, atr=float("nan"))
        self.assertEqual(sizing.notional, 0.0)
        self.assertIsNone(sizing.stop_price)
        self.assertEqual(sizing.binding, "no_stop_available")

    def test_a_healthy_atr_still_sizes_and_sets_a_stop(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        sizing = PositionSizer(cfg).size(nav=100_000.0, price=101.0, atr=2.0)
        self.assertGreater(sizing.notional, 0.0)
        self.assertIsNotNone(sizing.stop_price)

    def test_a_nan_mark_cannot_produce_a_stop_either(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        sizing = PositionSizer(cfg).size(nav=100_000.0, price=101.0, atr=float("inf"))
        self.assertEqual(sizing.notional, 0.0)

    def test_a_nan_price_is_never_accepted(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        sizing = PositionSizer(cfg).size(nav=100_000.0, price=float("nan"), atr=2.0)
        self.assertEqual(sizing.notional, 0.0)


class TestAnUnpriceablePositionCanStillBeExited(unittest.TestCase):
    def test_the_time_stop_fires_without_a_usable_mark(self) -> None:
        """The mark check used to sit in front of Rule 6, stranding the position."""
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        position = Position(pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=0.0, opened_ts_ms=1_000_000)
        signals = risk.protective_exits({PAIR: position}, {}, 1_800_000_000_000)
        self.assertTrue(
            any(s.meta.get("trigger") == "time_stop" for s in signals),
            f"a position with no mark must still reach the time stop, got {[s.meta for s in signals]}",
        )

    def test_the_execution_layer_sells_it_anyway(self) -> None:
        with scratch_dir() as d:
            client = _StatefulClient()
            engine = make_engine(d, client)
            self.addCleanup(engine.journal.close)
            engine.book.positions[PAIR] = Position(pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=0.0)
            engine._exit_long(
                ApprovedAction(pair=PAIR, action=EXIT_LONG, quantity=0.5, notional=30_000.0, reason="flatten"),
                int(time.time() * 1000),
                1,
            )
            self.assertEqual(len(client.orders_placed), 1, "an unpriceable position could not be sold")
            self.assertEqual(client.orders_placed[0][1], "SELL")


class TestNonFiniteNavCannotDisableTheHalts(unittest.TestCase):
    def test_a_nan_first_observation_leaves_the_kill_switch_armed(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        risk.observe(float("nan"), 1_800_000_000_000)
        self.assertTrue(math.isfinite(risk.peak_nav), "a NaN NAV poisoned the drawdown high-water mark")
        self.assertEqual(risk.peak_nav, 0.0)

    def test_a_nan_cannot_erase_a_real_high_water_mark(self) -> None:
        """The poisoning that mattered: a good peak, then a NaN, then the crash.

        `max(nan, x)` is NaN for every x and `nan <= 0` is False, so one non-finite
        mark used to make `peak_nav` NaN and the `if self.peak_nav > 0` guard false
        for the rest of the run -- the permanent kill switch simply stopped
        existing.
        """
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        risk.observe(100_000.0, 1_800_000_000_000)
        risk.observe(float("nan"), 1_800_000_000_000)
        self.assertTrue(math.isfinite(risk.peak_nav))
        self.assertAlmostEqual(risk.peak_nav, 100_000.0)
        risk.observe(1.0, 1_800_000_000_000)
        self.assertTrue(risk.halted, "a 99.999% drawdown did not trip the kill switch")

    def test_a_non_positive_nav_halts(self) -> None:
        cfg = make_config(Path(tempfile.gettempdir()))
        risk = RiskManager(cfg, PositionSizer(cfg))
        risk.observe(0.0, 1_800_000_000_000)
        self.assertTrue(risk.halted)


class TestMalformedStateCannotStopTheBoot(unittest.TestCase):
    def test_a_wrong_type_in_the_risk_state_is_ignored(self) -> None:
        with scratch_dir() as d:
            (d / "engine_state.json").write_text(
                json.dumps({"risk": {"peak_nav": "not-a-number", "last_exit_bar": {"ETH/USD": None}}}),
                encoding="utf-8",
            )
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            engine._load_risk_state()  # must not raise
            self.assertEqual(engine.risk.peak_nav, 0.0)
            self.assertEqual(engine.risk.last_exit_bar, {})

    def test_a_non_object_state_payload_is_ignored(self) -> None:
        with scratch_dir() as d:
            (d / "engine_state.json").write_text("[]", encoding="utf-8")
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            engine._load_risk_state()
            self.assertEqual(engine.risk.peak_nav, 0.0)

    def test_a_bad_position_row_does_not_cost_the_good_ones(self) -> None:
        with scratch_dir() as d:
            (d / "positions.json").write_text(
                json.dumps(
                    {
                        "positions": {
                            PAIR: {"quantity": None, "stop_price": "abc"},
                            "ETH/USD": {"quantity": 1.0, "avg_price": 2.0, "stop_price": 1.5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            self.assertTrue(engine.book.load())
            self.assertNotIn(PAIR, engine.book.positions, "an empty row was restored as a position")
            restored = engine.book.get("ETH/USD")
            self.assertIsNotNone(restored, "one bad row cost us the good one")
            self.assertEqual(restored.stop_price, 1.5)

    def test_a_non_numeric_stop_level_is_dropped_not_kept_as_a_string(self) -> None:
        with scratch_dir() as d:
            (d / "positions.json").write_text(
                json.dumps({"positions": {"ETH/USD": {"quantity": 1.0, "avg_price": 2.0, "stop_price": "abc"}}}),
                encoding="utf-8",
            )
            engine = make_engine(d, _StatefulClient())
            self.addCleanup(engine.journal.close)
            self.assertTrue(engine.book.load())
            position = engine.book.get("ETH/USD")
            self.assertIsNotNone(position)
            self.assertIsNone(position.stop_price, "a string stop level survived into the book")


class TestAHaltedRunStopsCleanlyInsteadOfRestarting(unittest.TestCase):
    """`Restart=always` used to restart a deliberately halted bot ~10 times.

    The halt is persisted, so a restart cannot recover anything. `run_live.py` now
    exits with a distinct status that the systemd unit declares a success, so the
    unit stops rather than thrashing.
    """

    def test_the_halt_status_is_distinct_from_success_and_failure(self) -> None:
        import run_live

        self.assertEqual(run_live.HALTED_EXIT_STATUS, 3)
        self.assertNotIn(run_live.HALTED_EXIT_STATUS, (0, 1))

    def test_the_unit_declares_that_status_a_clean_stop(self) -> None:
        import run_live

        unit = (Path(__file__).resolve().parent.parent / "deploy" / "roostoo-bot.service").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"SuccessExitStatus={run_live.HALTED_EXIT_STATUS}", unit)
        self.assertIn("Restart=always", unit)

    def test_a_halted_run_returns_the_halt_status(self) -> None:
        import run_live

        with scratch_dir() as d:
            env = d / ".env"
            env.write_text("ROOSTOO_MOCK=1\nJOURNAL_DIR=%s\nLOG_DIR=%s\n" % (d, d), encoding="utf-8")

            captured: dict = {}
            real_build = run_live.build_engine

            def build(cfg, seed, data_dir, interval):
                engine = real_build(cfg, seed, data_dir, interval)
                captured["engine"] = engine

                def fake_run(max_cycles=None):
                    engine.risk.halted = True
                    engine.risk.halt_reason = "drawdown 25.00% >= 20.00% of peak NAV"
                    return engine.stats

                engine.run = fake_run  # type: ignore[method-assign]
                return engine

            with mock.patch.object(run_live, "build_engine", build):
                status = run_live.main(["--mock", "--env", str(env), "--no-seed", "--log-level", "ERROR"])

            self.assertEqual(status, run_live.HALTED_EXIT_STATUS, "a halted run must not report success")
            captured["engine"].journal.close()


class HaltClient:
    """A usable venue whose sell orders are rejected, so a flatten cannot finish."""

    def __init__(self, hold: bool = True) -> None:
        self.orders: list = []
        #: Whether the venue still reports the coin. Reconciliation re-adopts from
        #: this, so clearing the local book is not enough to make it flat.
        self.hold = hold

    def sync_time(self) -> int:
        return 0

    def exchange_info(self) -> ExchangeInfo:
        return ExchangeInfo(is_running=True, initial_wallet={"USD": 100_000.0}, pairs={PAIR: trade_pair()})

    def ticker(self, pair: str | None = None) -> dict:
        return {PAIR: ticker(PAIR)}

    def balance(self) -> dict:
        # An *explicit* zero closes a position; a missing row never does.
        return {
            "USD": WalletBalance(asset="USD", free=100_000.0, locked=0.0),
            "BTC": WalletBalance(asset="BTC", free=0.5 if self.hold else 0.0, locked=0.0),
        }

    def pending_count(self) -> tuple[int, dict]:
        return 0, {}

    def place_order(self, *args, **kwargs):
        self.orders.append((args, kwargs))
        # The venue refuses to sell. The position must therefore survive.
        return OrderResult(
            pair=PAIR, side="SELL", order_type="MARKET", quantity=0.5,
            price=0.0, status="REJECTED", err_msg="insufficient balance",
        )

    def query_orders(self, **kwargs) -> list:
        return []

    def cancel_order(self, *args, **kwargs) -> list:
        return []

    def short_positions(self) -> list:
        return []


class TestHaltDoesNotAbandonOpenPositions(unittest.TestCase):
    """A halted engine must not walk away from an open position.

    `run_live` exits with HALTED_EXIT_STATUS and the systemd unit declares that
    status a clean stop with `RestartPreventExitStatus`, so a process that exits
    while a flatten is incomplete leaves the position with nothing managing it and
    nothing that will restart it. An unattended 14-day run cannot do that.
    """

    def _engine(self, directory: Path, hold: bool = True) -> TradingEngine:
        engine = TradingEngine(make_config(directory), client=HaltClient(hold=hold))
        self.addCleanup(engine.journal.close)
        engine.exchange_pairs = {PAIR: trade_pair()}
        engine._ready = True  # bootstrap is not under test here
        engine.risk.halted = True
        engine.risk.halt_reason = "drawdown 25.00% >= 20.00% of peak NAV"
        engine.book.positions[PAIR] = Position(
            pair=PAIR, quantity=0.5, avg_price=60_000.0, mark_price=61_000.0, stop_price=57_000.0
        )
        return engine

    def test_it_keeps_running_while_a_position_is_open(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d)
            engine.step()

            self.assertIn(PAIR, engine.book.positions, "the refused sell closed the position")
            self.assertFalse(
                engine._shutting_down,
                "the engine ended the run with a position still open and unsupervised",
            )

    def test_it_ends_the_run_once_the_book_is_flat(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d, hold=False)  # the venue reports an explicit zero
            engine.step()

            self.assertEqual(engine.book.held(), {})
            self.assertTrue(engine._shutting_down, "a flat book after a halt must end the run")

    def test_the_halt_is_announced_once_per_run(self) -> None:
        """The engine now loops while halted, so an unguarded journal.halt() would
        write one entry per cycle for the life of the process."""
        with scratch_dir() as d:
            engine = self._engine(d)
            calls: list = []
            real = engine.journal.halt
            engine.journal.halt = lambda *a, **k: (calls.append(1), real(*a, **k))[1]

            engine.step()
            engine.step()

            self.assertEqual(len(calls), 1, "the halt was journalled on every cycle")

    def test_incomplete_flattens_are_counted(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d)
            engine.step()
            engine.step()

            self.assertEqual(engine._halt_flatten_failures, 2)

    def test_the_failure_count_resets_once_the_book_is_flat(self) -> None:
        with scratch_dir() as d:
            engine = self._engine(d)
            engine.step()
            self.assertEqual(engine._halt_flatten_failures, 1)

            engine.client.hold = False  # the position finally leaves the venue
            engine.step()

            self.assertEqual(engine.book.held(), {})
            self.assertEqual(engine._halt_flatten_failures, 0)
            self.assertTrue(engine._shutting_down)


if __name__ == "__main__":
    unittest.main()
