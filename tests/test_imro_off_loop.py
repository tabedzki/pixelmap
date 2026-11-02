"""Downloads compute the IMRO table on a worker thread, not the event loop.

IMRO generation reads CSVs and checks the wiring of every selected electrode.
Even after indexing the wiring table it is tens of ms, and it slows several-
fold while a PDF render holds the GIL.  The three download callbacks (IMRO,
PDF, Kilosort) now snapshot the session's inputs on the loop and compute on a
worker, so other sessions are not held up.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading

import pytest

from pixelmap import backend
from pixelmap.types import Electrode
from pixelmap.utils import imro as imro_module
from tests.gui_helpers import select_preset

PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]
DOWNLOADS = ["generate_imro_content", "generate_pdf_content", "generate_kilosort_channelmap"]


# (button, patch target module, patched function, download callback)
ERROR_TARGETS = [
    pytest.param("download_imro_button", imro_module, "generate_imro_channelmap",
                 "generate_imro_content", id="imro"),
    pytest.param("download_pdf_button", backend, "probe_layout_pdf",
                 "generate_pdf_content", id="pdf"),
    pytest.param("download_kilosort_button", backend, "generate_kilosort_channelmap_dict",
                 "generate_kilosort_channelmap", id="kilosort"),
]


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Same output as the synchronous computation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_async_imro_matches_sync(gui, probe_type):
    select_preset(gui, probe_type)
    gui.generate_imro()
    expected = list(gui.imro_list)
    gui.imro_list = None
    gui._imro_memo = None  # compare a real async compute, not the memo

    _run(gui._generate_imro_async())

    assert gui.imro_list == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_async_imro_respects_legacy_header(gui, legacy):
    select_preset(gui, "2.0-4shanks")
    gui.legacy_imro_mode = legacy
    try:
        gui.generate_imro()
        expected = list(gui.imro_list)
        gui._imro_memo = None  # compare a real async compute, not the memo
        _run(gui._generate_imro_async())
        assert gui.imro_list == expected
    finally:
        gui.legacy_imro_mode = False


def test_imro_download_content(gui):
    select_preset(gui, "1.0")
    gui.generate_imro()
    header, *entries = gui.imro_list
    expected = f"({header[0]},{header[1]})" + "".join(
        "(" + " ".join(str(x) for x in e) + ")" for e in entries
    )
    buf = _run(gui.generate_imro_content())
    assert isinstance(buf, io.StringIO)
    assert buf.getvalue() == expected


def test_kilosort_download_content(gui):
    select_preset(gui, "2.0-4shanks")
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
    select_preset(gui, "1.0")
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
    select_preset(gui, "1.0")
    real = imro_module.generate_imro_channelmap
    threads = []

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(imro_module, "generate_imro_channelmap", spy)
    _run(gui._generate_imro_async())
    assert threads and threads[0] is not threading.current_thread()


def test_event_loop_runs_while_imro_computes(gui, monkeypatch):
    select_preset(gui, "1.0")
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
    select_preset(gui, "1.0")
    gui.generate_imro()
    expected = list(gui.imro_list)
    gui._imro_memo = None  # make the async call compute, not reuse this result
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


@pytest.mark.parametrize("button, module, attr, callback", ERROR_TARGETS)
def test_worker_error_reaches_caller(gui, monkeypatch, button, module, attr, callback):
    select_preset(gui, "1.0")

    def boom(*args, **kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(module, attr, boom)
    with pytest.raises(RuntimeError, match="worker failed"):
        _run(getattr(gui, callback)())


def test_sync_generate_imro_still_works(gui):
    """Other callers (and tests) use the synchronous method."""
    select_preset(gui, "2.0-1shank")
    gui.imro_list = None
    gui.generate_imro()
    assert gui.imro_list[0][1] == len(gui.imro_list) - 1


def test_inputs_snapshot_is_independent(gui):
    select_preset(gui, "1.0")
    inputs = gui._imro_inputs()
    before = tuple(inputs["selected"])
    gui.electrodes.clear_selection()
    assert inputs["selected"] == before


@pytest.mark.parametrize("button, module, attr, callback", ERROR_TARGETS)
def test_executor_download_errors_reach_session_handler(gui, monkeypatch, button, module, attr, callback):
    """Exercise FileDownload's scheduled server callback, including #75's handler.

    Raw Bokeh callbacks use the synchronous `guarded` decorator; downloads use
    Panel's async exception routing instead. Awaiting the executor future must
    preserve the exception and report it in the originating document context.
    """
    import weakref
    from unittest.mock import Mock

    import panel as pn
    from bokeh.document import Document
    from panel.io.state import set_curdoc

    from pixelmap import error_reporting

    select_preset(gui, "1.0")
    doc = Document()
    session = Mock()
    doc._session_context = weakref.ref(session)
    failure = RuntimeError(f"{attr} worker failed")
    worker_threads = []
    reports = []

    def boom(*args, **kwargs):
        worker_threads.append(threading.current_thread())
        raise failure

    monkeypatch.setattr(module, attr, boom)
    monkeypatch.setattr(
        error_reporting, "report_error",
        lambda exc, context: reports.append((exc, context, pn.state.curdoc)),
    )
    widget = getattr(gui, button)
    widget.data = None

    async def click():
        with set_curdoc(doc), pn.config.set(
            exception_handler=error_reporting.session_exception_handler
        ):
            widget._transfer()
            callback, = doc.session_callbacks
            await callback.callback()

    _run(click())

    assert worker_threads and worker_threads[0] is not threading.current_thread()
    assert reports == [(failure, "GUI callback", doc)]
    assert widget.data is None
