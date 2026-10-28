"""Downloads compute the IMRO table on a worker thread, not the event loop.

IMRO generation reads CSVs and checks the wiring of every selected electrode.
Even after indexing the wiring table it is tens of ms, and it slows several-
fold while a PDF render holds the GIL.  The three download callbacks (IMRO,
PDF, Kilosort) now snapshot the session's inputs on the loop and compute on a
worker, so other sessions are not held up.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading

import pytest

from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.types import Electrode
from pixelmap.utils import imro as imro_module

PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]
DOWNLOADS = ["generate_imro_content", "generate_pdf_content", "generate_kilosort_channelmap"]


@pytest.fixture(scope="module")
def gui():
    return ChannelmapGUI()


def _select_preset(gui, probe_type):
    gui.probe_type = probe_type
    gui.preset = next(p for p in gui.param.preset.objects if p)
    gui.apply_preset()


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Same output as the synchronous computation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_async_imro_matches_sync(gui, probe_type):
    _select_preset(gui, probe_type)
    gui.generate_imro()
    expected = list(gui.imro_list)
    gui.imro_list = None

    _run(gui._generate_imro_async())

    assert gui.imro_list == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_async_imro_respects_legacy_header(gui, legacy):
    _select_preset(gui, "2.0-4shanks")
    gui.legacy_imro_mode = legacy
    try:
        gui.generate_imro()
        expected = list(gui.imro_list)
        _run(gui._generate_imro_async())
        assert gui.imro_list == expected
    finally:
        gui.legacy_imro_mode = False


def test_imro_download_content(gui):
    _select_preset(gui, "1.0")
    gui.generate_imro()
    header, *entries = gui.imro_list
    expected = f"({header[0]},{header[1]})" + "".join(
        "(" + " ".join(str(x) for x in e) + ")" for e in entries
    )
    buf = _run(gui.generate_imro_content())
    assert isinstance(buf, io.StringIO)
    assert buf.getvalue() == expected


def test_kilosort_download_content(gui):
    _select_preset(gui, "2.0-4shanks")
    data = json.loads(_run(gui.generate_kilosort_channelmap()).getvalue())
    assert data["n_chan"] == 384
    assert len(data["chanMap"]) == len(data["xc"]) == len(data["yc"]) == len(data["kcoords"]) == 384


@pytest.mark.parametrize("name", DOWNLOADS)
def test_download_not_ready_returns_none(gui, monkeypatch, name):
    monkeypatch.setattr(gui, "ready_to_download", lambda: False)
    assert _run(getattr(gui, name)()) is None


@pytest.mark.parametrize("name", DOWNLOADS)
def test_download_callbacks_are_async(gui, name):
    """Panel awaits async callbacks instead of running them inline."""
    assert asyncio.iscoroutinefunction(getattr(gui, name))


@pytest.mark.parametrize(
    "button, prefix",
    [
        ("download_imro_button", b"("),
        ("download_pdf_button", b"%PDF"),
        ("download_kilosort_button", b"{"),
    ],
)
def test_downloads_through_panel_widget(gui, button, prefix):
    """Panel's own async transfer path delivers each file as a data URL."""
    import base64

    _select_preset(gui, "1.0")
    widget = getattr(gui, button)
    widget.data = None
    _run(widget._async_sync_data())
    assert widget.data.startswith("data:")
    payload = base64.b64decode(widget.data.split(",", 1)[1])
    assert payload.startswith(prefix)


# --------------------------------------------------------------------------
# Off the event loop
# --------------------------------------------------------------------------

def test_imro_computed_off_the_event_loop_thread(gui, monkeypatch):
    _select_preset(gui, "1.0")
    real = imro_module.generate_imro_channelmap
    threads = []

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", spy)
    _run(gui._generate_imro_async())
    assert threads and threads[0] is not threading.current_thread()


def test_event_loop_runs_while_imro_computes(gui, monkeypatch):
    _select_preset(gui, "1.0")
    real = imro_module.generate_imro_channelmap
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(10), "event loop never got to run"
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", blocked)

    async def main():
        task = asyncio.create_task(gui.generate_imro_content())
        while not started.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)  # would deadlock if the compute held the loop
        release.set()
        return await task

    assert _run(main()).getvalue().startswith("(")


def test_selection_change_during_compute_uses_snapshot(gui, monkeypatch):
    """Inputs are copied on the loop; editing the selection mid-compute must
    neither crash the worker ("set changed size") nor leak into the result."""
    _select_preset(gui, "1.0")
    gui.generate_imro()
    expected = list(gui.imro_list)
    real = imro_module.generate_imro_channelmap
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", blocked)

    async def main():
        task = asyncio.create_task(gui._generate_imro_async())
        while not started.is_set():
            await asyncio.sleep(0.01)
        gui.electrodes.clear_selection()
        gui.electrodes.select(Electrode(0, 0))
        release.set()
        await task

    _run(main())
    assert gui.imro_list == expected


def test_imro_error_reaches_caller(gui, monkeypatch):
    _select_preset(gui, "1.0")

    def boom(*args, **kwargs):
        raise RuntimeError("imro failed")

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", boom)
    with pytest.raises(RuntimeError, match="imro failed"):
        _run(gui.generate_imro_content())


def test_sync_generate_imro_still_works(gui):
    """Other callers (and tests) use the synchronous method."""
    _select_preset(gui, "2.0-1shank")
    gui.imro_list = None
    gui.generate_imro()
    assert gui.imro_list[0][1] == len(gui.imro_list) - 1


def test_inputs_snapshot_is_independent(gui):
    _select_preset(gui, "1.0")
    inputs = gui._imro_inputs()
    before = list(inputs["selected"])
    gui.electrodes.clear_selection()
    assert inputs["selected"] == before
