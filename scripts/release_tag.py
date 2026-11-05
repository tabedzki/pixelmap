"""Decide whether a pushed release tag should become the `latest` image.

Production follows `latest` through Watchtower, so only the highest final
release may have it. A pre-release, or a patch to an older minor, gets its own
version tags and leaves `latest` alone.

Usage: git tag --list 'v*' | release_tag.py v1.6.0 >> "$GITHUB_OUTPUT"
Exits non-zero, writing nothing to stdout, if the tag isn't a release tag.
"""

from __future__ import annotations

import re
import sys

_NUMBER = r"(0|[1-9]\d*)"
_FINAL = re.compile(rf"v{_NUMBER}\.{_NUMBER}\.{_NUMBER}")
_RELEASE = re.compile(rf"{_FINAL.pattern}(?:(?:a|b|rc)\d+)?(?:\.dev\d+)?")


def _final_version(tag: str) -> tuple[int, ...] | None:
    match = _FINAL.fullmatch(tag)
    return tuple(int(part) for part in match.groups()) if match else None


def is_latest(tag: str, existing: list[str]) -> bool:
    """Return whether `tag` is a final release at or above all of `existing`."""
    if not _RELEASE.fullmatch(tag):
        raise ValueError(
            f"{tag!r} is not a release tag: expected vX.Y.Z, with an optional "
            "aN, bN or rcN and an optional .devN"
        )
    version = _final_version(tag)
    if version is None:
        return False
    finals = (_final_version(other) for other in existing)
    return all(version >= other for other in finals if other is not None)


def main() -> int:
    try:
        latest = is_latest(sys.argv[1], sys.stdin.read().split())
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"is_latest={'true' if latest else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
