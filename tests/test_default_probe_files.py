"""Callers may omit ``wiring_file`` / ``positions_file``: the packaged CSVs are used."""

import inspect
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import pytest

from pixelmap import backend, constants
from pixelmap._probe_files import WIRING_MAPS_DIR, packaged_probe_file
from pixelmap.utils import imro

PROBE_TYPES = list(constants.WIRING_FILE_MAP)
PRESET = {
    "1.0": constants.SUPPORTED_1shank_PRESETS[0],
    "2.0-1shank": constants.SUPPORTED_1shank_PRESETS[0],
    "2.0-4shanks": constants.SUPPORTED_4shanks_PRESETS[0],
    "QuadBase": constants.SUPPORTED_QuadBase_PRESETS[0],
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_wiring_maps_dir_is_the_packaged_directory():
    assert WIRING_MAPS_DIR == Path(backend.__file__).resolve().parent / "wiring_maps"
    assert WIRING_MAPS_DIR.is_dir()


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_packaged_probe_file_matches_wiring_file_map(probe_type):
    positions, wiring = constants.WIRING_FILE_MAP[probe_type]
    assert packaged_probe_file(probe_type, "positions") == WIRING_MAPS_DIR / positions
    assert packaged_probe_file(probe_type, "wiring") == WIRING_MAPS_DIR / wiring
    assert packaged_probe_file(probe_type, "positions").is_file()
    assert packaged_probe_file(probe_type, "wiring").is_file()


@pytest.mark.parametrize("probe_type", [None, "", " ", "1.0 ", "NXT", "nhp", 0, ("1.0",), ["1.0"]])
def test_packaged_probe_file_rejects_unknown_probe_types(probe_type):
    with pytest.raises(ValueError, match="pass the file path explicitly"):
        packaged_probe_file(probe_type, "wiring")


@pytest.mark.parametrize("kind", [None, "", "Wiring", "position"])
def test_packaged_probe_file_rejects_unknown_kinds(kind):
    with pytest.raises(ValueError, match="kind"):
        packaged_probe_file("1.0", kind)


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_generate_imro_channelmap_defaults_to_the_packaged_wiring(probe_type):
    explicit = imro.generate_imro_channelmap(
        probe_type, PRESET[probe_type], wiring_file=packaged_probe_file(probe_type, "wiring")
    )
    assert imro.generate_imro_channelmap(probe_type, PRESET[probe_type]) == explicit
    assert (
        imro.generate_imro_channelmap(probe_type, PRESET[probe_type], wiring_file=None) == explicit
    )


def test_generate_imro_channelmap_explicit_wiring_file_wins(tmp_path):
    missing = tmp_path / "nowhere_wiring.csv"
    with pytest.raises(FileNotFoundError):
        imro.generate_imro_channelmap("1.0", PRESET["1.0"], wiring_file=missing)


@pytest.mark.parametrize("wiring_file", ["", "   "])
def test_generate_imro_channelmap_blank_path_is_not_a_default(wiring_file):
    """Only ``None`` selects the packaged file; a blank string is a caller bug."""
    with pytest.raises(FileNotFoundError):
        imro.generate_imro_channelmap("1.0", PRESET["1.0"], wiring_file=wiring_file)


def test_generate_imro_channelmap_unknown_probe_type_without_wiring_file():
    with pytest.raises(ValueError, match="NXT"):
        imro.generate_imro_channelmap("NXT", PRESET["1.0"], probe_subtype=0)


def test_plot_probe_layout_keeps_positional_order():
    params = list(inspect.signature(backend.plot_probe_layout).parameters)
    assert params[:5] == ["probe_type", "imro_list", "positions_file", "wiring_file", "title"]


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_plot_probe_layout_defaults_files_and_title(probe_type):
    imro_list = imro.generate_imro_channelmap(probe_type, PRESET[probe_type])

    backend.plot_probe_layout(probe_type, imro_list)

    titles = [ax.get_title() for ax in plt.gcf().axes if ax.get_title()]
    assert titles == [probe_type]


def test_plot_probe_layout_matches_explicit_files():
    imro_list = imro.generate_imro_channelmap("2.0-4shanks", PRESET["2.0-4shanks"])

    def facecolours():
        return [c.get_facecolor().tolist() for c in plt.gcf().axes[0].collections]

    backend.plot_probe_layout(
        "2.0-4shanks",
        imro_list,
        packaged_probe_file("2.0-4shanks", "positions"),
        packaged_probe_file("2.0-4shanks", "wiring"),
        "t",
    )
    explicit = facecolours()
    plt.close("all")
    backend.plot_probe_layout("2.0-4shanks", imro_list, title="t")
    assert facecolours() == explicit


@pytest.mark.parametrize("missing", ["positions_file", "wiring_file"])
def test_plot_probe_layout_explicit_file_wins(tmp_path, missing):
    imro_list = imro.generate_imro_channelmap("1.0", PRESET["1.0"])
    with pytest.raises(FileNotFoundError):
        backend.plot_probe_layout("1.0", imro_list, title="t", **{missing: tmp_path / "x.csv"})


def test_plot_probe_layout_unknown_probe_type_without_files():
    imro_list = imro.generate_imro_channelmap("1.0", PRESET["1.0"])
    with pytest.raises(ValueError, match="NXT"):
        backend.plot_probe_layout("NXT", imro_list)


def test_save_plot_uses_the_default_title(tmp_path):
    imro_list = imro.generate_imro_channelmap("1.0", PRESET["1.0"])
    backend.plot_probe_layout("1.0", imro_list, save_plot=True, saveDir=tmp_path)
    assert (tmp_path / "1.0.pdf").is_file()
