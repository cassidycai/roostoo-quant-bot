"""Seeding the live warm-up from the CSV directory.

Rules 2-3 need 48 closed bars before the first signal, so the warm-up is what
decides whether the bot can trade at all on day one. It used to be built from
``ROOSTOO_PAIRS``, which ships empty -- so by default nothing was ever seeded and
the bot started cold. These tests pin the directory scan that replaced it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_live  # noqa: E402

HEADER = "ts_ms,open,high,low,close,volume\n"


class ScratchDir:
    """Temporary directory that avoids mkdtemp's 0700 mode on Windows."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.gettempdir()) / f"seed_test_{id(self)}"
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, *exc: object) -> None:
        for child in self.path.glob("*"):
            child.unlink()
        self.path.rmdir()


class TestSeedDiscovery(unittest.TestCase):
    def test_discovers_every_pair_in_the_requested_interval(self) -> None:
        with ScratchDir() as d:
            (d / "BTC-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            (d / "ETH-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            (d / "ADA-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            got = run_live.discover_seed_paths(str(d), "30m")
            self.assertEqual(set(got), {"BTC/USD", "ETH/USD", "ADA/USD"})
            self.assertTrue(got["BTC/USD"].endswith("BTC-USD_30m.csv"))

    def test_ignores_other_intervals(self) -> None:
        with ScratchDir() as d:
            (d / "BTC-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            (d / "BTC-USD_4h.csv").write_text(HEADER, encoding="utf-8")
            got = run_live.discover_seed_paths(str(d), "4h")
            self.assertEqual(set(got), {"BTC/USD"})
            self.assertTrue(got["BTC/USD"].endswith("_4h.csv"))

    def test_ignores_synthetic_samples(self) -> None:
        """``data/sample_*.csv`` are test fixtures and stay in git; they are not
        history and must never be glued onto a live book."""
        with ScratchDir() as d:
            (d / "BTC-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            (d / "sample_BTC-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            (d / "sample_ETH-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            got = run_live.discover_seed_paths(str(d), "30m")
            self.assertEqual(set(got), {"BTC/USD"})

    def test_empty_directory_yields_nothing(self) -> None:
        with ScratchDir() as d:
            self.assertEqual(run_live.discover_seed_paths(str(d), "30m"), {})

    def test_missing_directory_yields_nothing(self) -> None:
        """A cold start must not raise; it just has no history to load."""
        self.assertEqual(run_live.discover_seed_paths("/nonexistent/data", "30m"), {})

    def test_pair_keys_match_the_venue_format(self) -> None:
        """The venue quotes BASE/QUOTE with a slash; a key in the wrong format is
        silently ignored by seed_history, which is the bug this scan fixes."""
        with ScratchDir() as d:
            (d / "AVAX-USD_30m.csv").write_text(HEADER, encoding="utf-8")
            got = run_live.discover_seed_paths(str(d), "30m")
            self.assertIn("AVAX/USD", got)
            self.assertNotIn("AVAX-USD", got)


if __name__ == "__main__":
    unittest.main()
