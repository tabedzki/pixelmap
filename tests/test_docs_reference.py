"""The docs must describe the package as it is.

Each check catches a kind of drift that has already happened:
example paths that only resolve from inside the package directory, absolute
user paths in notebooks or docs, and a docstring that lists arguments the
function doesn't take.
"""

import inspect
import re
import subprocess
from pathlib import Path

import pytest

from pixelmap import constants
from pixelmap.utils.imro import generate_imro_channelmap

REPO = Path(__file__).resolve().parent.parent
WIRING_DIR = REPO / "pixelmap" / "wiring_maps"
DOC_FILES = sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]


def _doc_ids():
    return [p.relative_to(REPO).as_posix() for p in DOC_FILES]


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_ids())
def test_example_wiring_files_exist(doc):
    """Wiring CSVs named in the docs ship with the package."""
    names = set(re.findall(r'"([\w.-]+_wiring\.csv)"', doc.read_text()))
    missing = sorted(n for n in names if not (WIRING_DIR / n).exists())
    assert not missing, f"{doc.name} names wiring files that don't exist: {missing}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_ids())
def test_examples_do_not_use_cwd_relative_wiring_paths(doc):
    """`"wiring_maps/..."` only resolves from inside the package directory."""
    hits = re.findall(r'"wiring_maps/[^"]*"', doc.read_text())
    assert not hits, f"{doc.name} uses working-directory-relative paths: {hits}"


def test_generate_imro_channelmap_docstring_matches_signature():
    doc = inspect.getdoc(generate_imro_channelmap)
    args_block = doc.split("Args:", 1)[1].split("Returns:", 1)[0]
    documented = re.findall(r"^\s{4}(\w+):", args_block, flags=re.MULTILINE)
    params = list(inspect.signature(generate_imro_channelmap).parameters)
    assert documented == params


def test_generate_imro_channelmap_docstring_describes_the_list_it_returns():
    returns = inspect.getdoc(generate_imro_channelmap).split("Returns:", 1)[1]
    assert "string" not in returns.lower()
    assert "list" in returns.lower()


def test_documented_probe_types_are_supported():
    """The probe types listed for `probe_type` all have wiring files."""
    doc = inspect.getdoc(generate_imro_channelmap)
    line = next(ln for ln in doc.splitlines() if ln.strip().startswith("probe_type:"))
    listed = re.findall(r'"([^"]+)"', line)
    assert listed, "probe_type should list the supported values"
    assert set(listed) == set(constants.WIRING_FILE_MAP)


def test_tracked_notebooks_and_docs_have_no_absolute_user_paths():
    """Shared examples must not depend on a contributor's home directory."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO, text=True
    ).split("\0")
    paths = [
        REPO / name
        for name in tracked
        if name and (
            Path(name).suffix in {".ipynb", ".md", ".rst"}
            or (name.startswith("docs/") and Path(name).suffix in {".py", ".txt"})
        )
    ]
    assert paths
    for path in paths:
        # Covers plain text and JSON-escaped Windows paths in notebook cells.
        assert not re.search(r"/Users/|[A-Za-z]:\\+Users", path.read_text()), (
            f"{path.relative_to(REPO)} contains an absolute user path"
        )
