#!/usr/bin/env python3
"""Guard against the encoding damage that Windows PowerShell quietly causes.

This repository is written on Windows and read on Linux and GitHub. A PowerShell
pipeline such as ``Get-Content -Raw x.md | Set-Content x.md -Encoding utf8``
decodes UTF-8 as the machine's ANSI codepage (cp936/GBK on a Chinese Windows) and
re-encodes the result, so every em dash becomes the two-character sequence
U+9225 U+003F, a BOM appears at the top of the file, and no tool reports an error.
That happened once; this script is what stops it happening again.

Two checks, both fatal:

* **No UTF-8 BOM.** Markdown, YAML, Python and ``.example`` files in this repo
  have never carried one, and Git for Windows tooling does not add them.
* **No mojibake.** Characters are exhausted against an allowlist of scripts the
  docs legitimately use (ASCII, Latin-1 punctuation, Greek for the sigma/delta
  maths, general punctuation, arrows, mathematical operators, box drawing, CJK and
  the fullwidth forms). Anything outside those ranges -- a stray U+9225, a Private
  Use Area codepoint -- is a double-encoding artefact.

The guard deliberately does **not** police which language the team writes in; it
catches byte-level damage, and the CJK ranges stay allowed for that reason.

Run standalone, from the pre-commit hook, or from CI::

    python scripts/check_encoding.py            # whole repository
    python scripts/check_encoding.py README.md  # specific paths
"""

from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path

#: Extensions whose contents are text this repository authors itself.
TEXT_SUFFIXES = {
    ".md", ".py", ".yml", ".yaml", ".toml", ".json", ".csv", ".txt",
    ".example", ".service", ".ps1", ".sh", ".cfg", ".ini",
}

BOM = b"\xef\xbb\xbf"

#: Codepoint ranges the documents are allowed to contain.
#:
#: Deliberately generous inside each range and strict about the ranges: the point
#: is to catch double-encoding, not to police which words the team may write.
ALLOWED_RANGES = (
    (0x0000, 0x007F),  # ASCII
    (0x00A0, 0x00FF),  # Latin-1 supplement: plus-minus, multiply, divide
    (0x0391, 0x03C9),  # Greek: delta, sigma
    (0x2010, 0x203A),  # general punctuation: en/em dash, curly quotes, ellipsis
    (0x20AC, 0x20AC),  # euro sign
    (0x2100, 0x2138),  # letterlike symbols: degree Celsius, trademark
    (0x2190, 0x21FF),  # arrows
    (0x2200, 0x22FF),  # mathematical operators: minus, less/greater, not-equal
    (0x2460, 0x24FF),  # enclosed alphanumerics
    (0x2500, 0x257F),  # box drawing
    (0x25A0, 0x25FF),  # geometric shapes
    (0x2600, 0x27BF),  # misc symbols and dingbats: check mark
    (0x3000, 0x303F),  # CJK punctuation
    (0x3040, 0x30FF),  # hiragana / katakana (used as literal examples)
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xFF00, 0xFFEF),  # fullwidth forms
)


def _allowed(ch: str) -> bool:
    cp = ord(ch)
    if cp in KNOWN_MOJIBAKE:
        return False
    return any(low <= cp <= high for low, high in ALLOWED_RANGES)


#: Codepoints that only ever appeared in this repository as the result of a
#: cp936 round trip, harvested from the real damage. An explicit blocklist rather
#: than narrower CJK ranges on purpose: the docs are authored by a multilingual
#: team, so tightening the ideograph range would flag real text.
#:
#: These specific characters are what an em dash, en dash, right arrow and
#: less-than-or-equal sign turn into when their UTF-8 bytes are read as GBK (the
#: third byte is an incomplete sequence, which shifts the following bytes and
#: yields CJK lookalikes).
#:
#: The codepoints are written as numbers rather than spelled out, so that this
#: file contains no CJK characters of its own.
KNOWN_MOJIBAKE = frozenset(
    {
        0x20AC,  # euro sign -- appeared in the damaged README
        0x2103,  # degree Celsius
        0x3221,  # parenthesized ideograph two
        0x300D,  # right corner bracket
        0xFF45,  # fullwidth latin small letter e
        0xFFE0,  # fullwidth cent sign
        0x9225,  # the em dash artefact, the most common one
        0x9286,
        0x951B,
        0x93C8,
        0x93C9,
        0x9428,
        0x9429,
        0x95BD,
        0x95C0,
        0x95C7,
        0x922D,
        0x922E,
        0x934A,
        0x934F,
        0x9350,
        0x9351,
        0x9358,
        0x9359,
        0x935A,
        0x9365,
        0x9366,
        0x93B5,
        0x93B7,
        0x93B9,
        0x93BA,
        0x93BB,
        0x93C1,
        0x93C2,
        0x93C3,
        0x93C4,
        0x93CD,
        0x87FD,  # the sigma artefact
        *range(0xE000, 0xF900),  # private use area, written by the bad encoder
    }
)


def _label(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return "<unnamed>"


#: Files that legitimately quote the artefacts they detect, and so must be
#: exempt from their own rule. `scan_secrets.py` needs the same exemption for its
#: pattern list. Adding an entry here is a deliberate decision: it is a place a
#: real re-encoding bug could hide.
SELF_EXEMPT = frozenset({"scripts/check_encoding.py"})


def _skip(path: Path) -> bool:
    """Is this file exempt from the guard?"""
    return path.as_posix() in SELF_EXEMPT


def check_file(path: Path) -> list[str]:
    """Return human-readable problems for one file (empty means clean)."""
    problems: list[str] = []
    if _skip(path):
        return problems
    raw = path.read_bytes()

    if raw.startswith(BOM):
        problems.append("starts with a UTF-8 BOM (this repo's files do not carry one)")

    body = raw[3:] if raw.startswith(BOM) else raw
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"is not valid UTF-8: {exc}")
        return problems

    for line_no, line in enumerate(text.splitlines(), start=1):
        offenders = [ch for ch in line if not _allowed(ch)]
        if offenders:
            seen: list[str] = []
            for ch in offenders:
                shown = f"U+{ord(ch):04X} {ch!r} ({_label(ch)})"
                if shown not in seen:
                    seen.append(shown)
            problems.append(f"line {line_no}: unexpected character(s) {', '.join(seen[:4])}")
            if len(problems) >= 6:
                problems.append("... further problems suppressed")
                break
    return problems


def tracked_text_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    listing = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
    return [Path(p) for p in listing if Path(p).suffix.lower() in TEXT_SUFFIXES]


def main(argv: list[str]) -> int:
    files = tracked_text_files(argv)
    if not files:
        print("no text files to check")
        return 0

    damaged = 0
    for path in files:
        if not path.is_file():
            continue
        problems = check_file(path)
        if problems:
            damaged += 1
            print(f"{path}:")
            for problem in problems:
                print(f"    {problem}")

    if damaged:
        print()
        print(f"FAILED: {damaged} of {len(files)} file(s) look re-encoded.")
        print("A PowerShell pipe such as `Get-Content x | Set-Content x` does this.")
        print("Rewrite the file from its original bytes and apply edits again.")
        return 1

    print(f"encoding OK: {len(files)} text file(s) are BOM-free UTF-8 with no mojibake")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
