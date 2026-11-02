"""Repeat downloads reuse work instead of redoing it.

A user who downloads the IMRO, then the PDF, then the Kilosort map used to
recompute the IMRO table three times (17-90 ms each, GIL-bound) and re-read
the probe's wiring and positions CSVs on every download.  Each session now
remembers its last IMRO, keyed on everything the computation reads, and the
static probe CSVs are parsed once per path per process.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pytest

from pixelmap import backend
from pixelmap.constants import WIRING_FILE_MAP
from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.utils import imro as imro_module
from tests.gui_helpers import select_preset

DOWNLOADS = ["generate_imro_content", "generate_pdf_content", "generate_kilosort_channelmap"]
WIRING_DIR = Path(backend.__file__).parent / "wiring_maps"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def computes(monkeypatch):
    """Count IMRO computations (the expensive call behind every download)."""
    calls = []
    real = imro_module.generate_imro_channelmap

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", spy)
    return calls


@pytest.fixture
def fast_pdf(monkeypatch):
    """Skip the ~1 s render; these tests are about the IMRO feeding it."""
    rendered = []

    def render(probe_type, imro_list, *args, **kwargs):
        rendered.append(list(imro_list))
        return BytesIO(b"%PDF")

    monkeypatch.setattr(backend, "probe_layout_pdf", render)
    return rendered


@pytest.fixture
def restore_settings(gui):
    """Undo the setting changes a test makes on the module-scoped GUI."""
    saved = (gui.reference_id, gui.ap_gain_input.value, gui.lf_gain_input.value,
             gui.hardware_hp_filter_on, gui.legacy_imro_mode)
    yield
    select_preset(gui, "2.0-4shanks")
    gui.probe_subtype = gui.param.probe_subtype.objects[0]
    (gui.reference_id, gui.ap_gain_input.value, gui.lf_gain_input.value,
     gui.hardware_hp_filter_on, gui.legacy_imro_mode) = saved


# --------------------------------------------------------------------------
# IMRO memo
# --------------------------------------------------------------------------

def test_imro_pdf_kilosort_compute_the_imro_once(gui, computes, fast_pdf):
    select_preset(gui, "2.0-4shanks")
    imro_text = _run(gui.generate_imro_content()).getvalue()
    _run(gui.generate_pdf_content())
    kilosort = _run(gui.generate_kilosort_channelmap()).getvalue()

    assert len(computes) == 1
    assert '"n_chan": 384' in kilosort
    # The cached table is the one the IMRO download wrote out.
    header = fast_pdf[0][0]
    assert imro_text.startswith(f"({header[0]},{header[1]})")


def test_sync_generate_imro_shares_the_memo(gui, computes):
    select_preset(gui, "1.0")
    gui.generate_imro()
    _run(gui.generate_imro_content())
    gui.generate_imro()
    assert len(computes) == 1


def _change_selection(gui):
    """Another complete selection (a partial one isn't downloadable)."""
    gui.preset = [p for p in gui.param.preset.objects if p][1]
    gui.apply_preset()


def _set(attr, value):
    def change(gui):
        setattr(gui, attr, value)
    return change


def _set_widget(widget, value):
    def change(gui):
        getattr(gui, widget).value = value
    return change


# (change, whether the 4-shank IMRO text changes; gains and the hardware
# filter only appear in 1.0-format tables, but must still miss the memo)
CHANGES = [
    pytest.param(_change_selection, True, id="selection"),
    pytest.param(_set("reference_id", "Tip shank 0"), True, id="reference"),
    pytest.param(_set("probe_subtype", "NP2014"), True, id="probe_subtype"),
    pytest.param(_set_widget("ap_gain_input", 1000), False, id="ap_gain"),
    pytest.param(_set_widget("lf_gain_input", 125), False, id="lf_gain"),
    pytest.param(_set("hardware_hp_filter_on", 0), False, id="hp_filter"),
    pytest.param(_set("legacy_imro_mode", True), True, id="legacy"),
]


@pytest.mark.parametrize("change, changes_output", CHANGES)
def test_any_input_change_recomputes(gui, computes, restore_settings, change, changes_output):
    select_preset(gui, "2.0-4shanks")
    first = _run(gui.generate_imro_content()).getvalue()
    _run(gui.generate_imro_content())
    assert len(computes) == 1  # unchanged: a hit

    change(gui)
    second = _run(gui.generate_imro_content()).getvalue()

    assert len(computes) == 2  # changed: a miss
    assert (second != first) == changes_output


def test_int_and_float_gain_are_different_keys(gui, computes, restore_settings):
    """500 == 500.0 and they hash alike, but the 1.0 IMRO writes them out
    differently ("500" vs "500.0"), so the memo must not conflate them."""
    select_preset(gui, "1.0")
    gui.ap_gain_input.value = 500
    as_int = _run(gui.generate_imro_content()).getvalue()
    _run(gui.generate_imro_content())
    assert len(computes) == 1

    gui.ap_gain_input.value = 500.0
    as_float = _run(gui.generate_imro_content()).getvalue()
    assert len(computes) == 2
    assert type(computes[1]["ap_gain"]) is float
    assert " 500 " in as_int and " 500.0 " in as_float


def test_imro_inputs_are_hashable(gui):
    select_preset(gui, "2.0-4shanks")
    inputs = gui._imro_inputs()
    for value in inputs.values():
        hash(value)  # raises TypeError for an unhashable field


def test_selection_order_does_not_matter(gui):
    """The memo key sorts the selection, which is sound only because the IMRO
    is sorted by (unique) channel whatever order the electrodes come in."""
    select_preset(gui, "QuadBase")
    inputs = gui._imro_inputs()
    selected = list(inputs["selected"])
    forward = imro_module.generate_imro_channelmap(
        probe_type=inputs["probe_type"], custom_electrodes=pd.DataFrame(selected).to_numpy(),
        wiring_file=inputs["wiring_file"], probe_subtype=inputs["probe_subtype"],
    )
    backward = imro_module.generate_imro_channelmap(
        probe_type=inputs["probe_type"], custom_electrodes=pd.DataFrame(selected[::-1]).to_numpy(),
        wiring_file=inputs["wiring_file"], probe_subtype=inputs["probe_subtype"],
    )
    assert forward == backward


def test_callers_cannot_corrupt_the_memo(gui, computes):
    select_preset(gui, "1.0")
    first = _run(gui._generate_imro_async())
    expected = list(first)
    first[0] = ("corrupted", 0)
    first.append((0, 0, 0, 0, 0, 0))
    gui.imro_list.clear()

    again = _run(gui._generate_imro_async())

    assert len(computes) == 1
    assert again == expected


def test_sessions_do_not_share_memos(computes, fast_pdf):
    one, four = ChannelmapGUI(), ChannelmapGUI()
    select_preset(one, "1.0")
    select_preset(four, "2.0-4shanks")

    contents = []
    for _ in range(2):
        for session in (one, four):
            contents.append(_run(session.generate_imro_content()).getvalue())

    assert len(computes) == 2
    assert contents[0] == contents[2] != contents[1] == contents[3]
    assert contents[0].startswith("(NP1000,") and contents[1].startswith("(NP2010,")


def test_empty_selection_does_not_compute_or_clobber_the_memo(gui, computes):
    select_preset(gui, "2.0-4shanks")
    first = _run(gui.generate_imro_content()).getvalue()

    saved = set(gui.electrodes.selected)
    gui.electrodes.clear_selection()
    assert not gui.ready_to_download()
    for name in DOWNLOADS:
        assert _run(getattr(gui, name)()) is None

    for electrode in saved:
        gui.electrodes.select(electrode)
    again = _run(gui.generate_imro_content()).getvalue()

    assert len(computes) == 1
    assert again == first


def test_switching_probe_type_misses(gui, computes):
    select_preset(gui, "2.0-4shanks")
    four = _run(gui.generate_imro_content()).getvalue()

    select_preset(gui, "1.0")
    one = _run(gui.generate_imro_content()).getvalue()
    _run(gui.generate_imro_content())
    assert len(computes) == 2
    assert one != four

    select_preset(gui, "2.0-4shanks")  # one-entry memo: back is a miss again
    assert _run(gui.generate_imro_content()).getvalue() == four
    assert len(computes) == 3


def test_failed_compute_is_not_memoised(gui, monkeypatch):
    select_preset(gui, "1.0")
    real = imro_module.generate_imro_channelmap
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", flaky)
    with pytest.raises(RuntimeError):
        _run(gui.generate_imro_content())
    assert _run(gui.generate_imro_content()).getvalue().startswith("(")
    assert len(calls) == 2


# --------------------------------------------------------------------------
# Probe CSVs
# --------------------------------------------------------------------------

@pytest.fixture
def csv_reads(monkeypatch):
    """Count pd.read_csv calls per resolved path, starting from a cold cache."""
    backend._read_csv_cached.cache_clear()
    reads = Counter()
    real = pd.read_csv

    def spy(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            reads[Path(path).resolve()] += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    yield reads
    backend._read_csv_cached.cache_clear()


@pytest.mark.parametrize("probe_type", ["1.0", "2.0-4shanks"])
def test_downloads_read_each_csv_once(gui, csv_reads, probe_type):
    select_preset(gui, probe_type)
    for _ in range(2):
        gui._imro_memo = None  # force the IMRO path to read the wiring again
        for name in DOWNLOADS:
            assert _run(getattr(gui, name)()) is not None

    assert set(csv_reads) == {gui.positions_file.resolve(), gui.wiring_file.resolve()}
    assert set(csv_reads.values()) == {1}


def test_path_spellings_share_one_read(csv_reads):
    path = WIRING_DIR / WIRING_FILE_MAP["1.0"][1]
    relative = os.path.relpath(path)
    dotted = WIRING_DIR / ".." / "wiring_maps" / path.name
    frames = [backend._read_probe_csv(p) for p in (path, str(path), relative, dotted)]
    assert sum(csv_reads.values()) == 1
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frame, frames[0])


def test_cached_frames_are_private_copies(csv_reads):
    path = WIRING_DIR / WIRING_FILE_MAP["2.0-4shanks"][0]
    first = backend._read_probe_csv(path)
    expected = first.copy()

    first.iloc[0, 0] = -999
    first["x"] = 0
    first.drop(columns="y", inplace=True)
    first.columns = ["a"] * len(first.columns)

    second = backend._read_probe_csv(path)
    assert second is not first
    pd.testing.assert_frame_equal(second, expected)


def test_public_functions_cannot_corrupt_the_cache(gui, csv_reads):
    """Every public consumer gets a frame of its own."""
    select_preset(gui, "2.0-4shanks")
    gui.generate_imro()
    before_pos = backend._read_probe_csv(gui.positions_file)
    before_wiring = backend._read_probe_csv(gui.wiring_file)

    backend.generate_kilosort_channelmap_dict(gui.imro_list, gui.positions_file)
    backend.probe_layout_pdf(gui.probe_type, gui.imro_list, gui.positions_file, gui.wiring_file, "t")
    imro_module.generate_imro_channelmap("2.0-4shanks", layout_preset="tips_all",
                                         wiring_file=gui.wiring_file)

    pd.testing.assert_frame_equal(backend._read_probe_csv(gui.positions_file), before_pos)
    pd.testing.assert_frame_equal(backend._read_probe_csv(gui.wiring_file), before_wiring)


def test_rewritten_csv_is_read_again(tmp_path, csv_reads):
    path = tmp_path / "positions.csv"
    path.write_text("shank,electrode,x,y\n0,0,1,2\n")
    assert backend._read_probe_csv(path)["x"].tolist() == [1]
    path.write_text("shank,electrode,x,y\n0,0,30,40\n0,1,5,6\n")
    os.utime(path, ns=(1, 1))  # different mtime even on coarse-clock filesystems
    assert backend._read_probe_csv(path)["x"].tolist() == [30, 5]
    assert csv_reads[path.resolve()] == 2


def test_public_functions_still_accept_paths_and_buffers(gui):
    select_preset(gui, "1.0")
    gui.generate_imro()
    expected = backend.generate_kilosort_channelmap_dict(gui.imro_list, gui.positions_file)
    as_str = backend.generate_kilosort_channelmap_dict(gui.imro_list, str(gui.positions_file))
    as_buffer = backend.generate_kilosort_channelmap_dict(
        gui.imro_list, StringIO(gui.positions_file.read_text())
    )
    for got in (as_str, as_buffer):
        assert got["n_chan"] == expected["n_chan"]
        assert (got["xc"] == expected["xc"]).all() and (got["yc"] == expected["yc"]).all()

    wiring = StringIO(gui.wiring_file.read_text())
    selected = [(e.shank_id, e.electrode_id) for e in sorted(
        gui.electrodes.selected, key=lambda e: (e.shank_id, e.electrode_id))]
    assert imro_module.generate_imro_channelmap(
        "1.0", custom_electrodes=pd.DataFrame(selected).to_numpy(), wiring_file=wiring,
        probe_subtype=gui.probe_subtype,
    )[0][1] == len(selected)


def test_missing_csv_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        backend._read_probe_csv(tmp_path / "nope.csv")
