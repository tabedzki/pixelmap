"""Tests for how a pushed git tag becomes a published release.

Production follows the `latest` image through Watchtower, so anything that
moves `latest` by mistake deploys by mistake. The config checks pin the rules
that a YAML linter can't see.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
SCRIPT = REPO / "scripts" / "release_tag.py"


def _load_release_tag():
    spec = importlib.util.spec_from_file_location("release_tag", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("tag", "existing"),
    [
        # The only release so far, including when git lists nothing.
        ("v1.0.0", []),
        ("v1.0.0", ["v1.0.0"]),
        ("v0.0.0", ["v0.0.0"]),
        # Newest final release.
        ("v1.6.0", ["v1.5.0", "v1.5.1", "v1.6.0"]),
        # Numeric, not string, ordering.
        ("v1.10.0", ["v1.9.0", "v1.10.0"]),
        ("v10.0.0", ["v9.9.9", "v10.0.0"]),
        # Newer pre-releases and malformed tags don't outrank a final release.
        ("v1.6.0", ["v1.6.0", "v1.7.0rc1", "v2.0.0.dev1", "v9.9.9-backport"]),
    ],
)
def test_newest_final_release_is_latest(tag, existing):
    assert _load_release_tag().is_latest(tag, existing) is True


@pytest.mark.parametrize(
    ("tag", "existing"),
    [
        # A patch to an old minor must not move prod backwards.
        ("v1.5.2", ["v1.5.2", "v1.6.0"]),
        ("v1.9.0", ["v1.9.0", "v1.10.0"]),
        # Re-pushing an older tag.
        ("v1.4.2", ["v1.4.2", "v1.4.3"]),
        # Pre-releases never, even when they are the highest version.
        ("v1.6.0rc1", ["v1.5.0", "v1.6.0rc1"]),
        ("v1.6.0a1", ["v1.6.0a1"]),
        ("v1.6.0b2", ["v1.6.0b2"]),
        ("v1.6.0.dev1", ["v1.6.0.dev1"]),
        ("v1.6.0rc1.dev1", ["v1.6.0rc1.dev1"]),
    ],
)
def test_other_releases_are_not_latest(tag, existing):
    assert _load_release_tag().is_latest(tag, existing) is False


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "v",
        "1.6.0",
        "v1.6",
        "v1.6.0.0",
        "v01.6.0",
        "v1.6.0-backport",
        "v1.6.0-rc1",
        "v1.6.0.post1",
        "v1.6.0rc",
        "version-1",
        "V1.6.0",
        "v1.6.0\n",
    ],
)
def test_tags_we_cannot_version_are_rejected(tag):
    with pytest.raises(ValueError, match="release tag"):
        _load_release_tag().is_latest(tag, [])


def test_cli_writes_a_github_output_line():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "v1.6.0"],
        input="v1.5.0\nv1.6.0\nv1.7.0rc1\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "is_latest=true\n"


def test_cli_fails_on_a_bad_tag_without_writing_output():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "v1.6.0-backport"],
        input="",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "v1.6.0-backport" in result.stderr


def test_publish_builds_each_release_once():
    assert not re.search(r"^  release:", _read(WORKFLOWS / "docker-publish.yml"), re.MULTILINE)


def test_latest_comes_only_from_the_release_tag_check():
    publish = _read(WORKFLOWS / "docker-publish.yml")
    # latest=auto would add `latest` to other tag rules behind the check's back.
    assert "latest=false" in publish
    assert re.findall(r"type=raw,value=latest,enable=(.+)", publish) == [
        "${{ steps.release.outputs.is_latest == 'true' }}"
    ]


def test_publish_waits_for_tests():
    publish = _read(WORKFLOWS / "docker-publish.yml")
    assert "uses: ./.github/workflows/tests.yml" in publish
    assert re.search(r"^    needs: tests$", publish, re.MULTILINE)
    assert "workflow_call:" in _read(WORKFLOWS / "tests.yml")


def test_nothing_targets_the_retired_dev_branch():
    for workflow in WORKFLOWS.glob("*.yml"):
        assert not re.search(r"\bdev\b", _branch_filters(_read(workflow))), workflow


def _branch_filters(workflow: str) -> str:
    return "\n".join(re.findall(r"branches:.*(?:\n\s+- .*)*", workflow))


@pytest.mark.parametrize("workflow", ["docker-publish.yml", "tests.yml"])
def test_checkouts_fetch_tags(workflow):
    # setuptools-scm reads the version from tags; a shallow clone reports 0.1.dev1.
    text = _read(WORKFLOWS / workflow)
    checkouts = text.count("uses: actions/checkout@")
    assert checkouts > 0
    assert text.count("fetch-depth: 0") == checkouts


def test_ci_and_image_use_the_same_uv():
    image = re.findall(r"astral-sh/uv:(\S+)", _read(REPO / "Dockerfile"))
    assert len(image) == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+", image[0])
    for workflow in ("docker-publish.yml", "tests.yml"):
        text = _read(WORKFLOWS / workflow)
        setups = text.count("uses: astral-sh/setup-uv@")
        assert setups > 0
        assert re.findall(r"^\s+version: '(.+?)'", text, re.MULTILINE) == image * setups


def test_base_image_is_pinned():
    bases = re.findall(r"^FROM (\S+)", _read(REPO / "Dockerfile"), re.MULTILINE)
    assert bases
    for base in bases:
        assert ":" in base
        assert not base.endswith(":latest")


def test_python_version_has_one_source():
    assert re.fullmatch(r"\d+\.\d+\n", _read(REPO / ".python-version"))
    assert "UV_PYTHON" not in _read(REPO / "Dockerfile")
    assert "UV_PYTHON" not in _read(WORKFLOWS / "tests.yml")


def test_version_comes_from_git_tags():
    pyproject = tomllib.loads(_read(REPO / "pyproject.toml"))
    assert any(r.startswith("setuptools-scm") for r in pyproject["build-system"]["requires"])
    assert pyproject["project"]["dynamic"] == ["version"]
    assert "dynamic" not in pyproject["tool"]["setuptools"]
    assert "setuptools_scm" in pyproject["tool"]


def test_image_build_does_not_need_git_history():
    # The workflow passes the version in, so .git stays out of the image.
    assert ".git" in _read(REPO / ".dockerignore").split()
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PIXELMAP" in _read(REPO / "Dockerfile")
