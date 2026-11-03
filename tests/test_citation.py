"""CITATION.cff must describe the version the package reports."""

import re
from pathlib import Path

import pixelmap

CITATION = Path(__file__).resolve().parents[1] / "CITATION.cff"
CONCEPT_DOI = "10.5281/zenodo.22850327"  # resolves to the latest archived version


def _top_level(key):
    match = re.search(rf"^{key}:\s*'?([^'\n]+?)'?\s*$", CITATION.read_text(), flags=re.MULTILINE)
    assert match, f"CITATION.cff has no top-level {key!r}"
    return match.group(1)


def test_citation_version_matches_package_version():
    assert _top_level("version") == pixelmap.__version__, (
        "Bump CITATION.cff `version` and `date-released` together with pixelmap.__version__"
    )


def test_citation_date_released_is_an_iso_date():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", _top_level("date-released"))


def test_citation_doi_does_not_pin_a_single_release():
    """The concept DOI stays correct across releases, so a version bump only touches
    ``version`` and ``date-released``. A per-release DOI would go stale at every bump."""
    assert _top_level("doi") == CONCEPT_DOI
