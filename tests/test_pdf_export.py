"""Regression tests: a PDF download must not stall the server.

Production watchdog logs showed the single Bokeh event loop blocked for 5-6 s
inside ``generate_pdf_content``: once in the legend's ``loc="best"`` search
(which scores candidate spots against every electrode patch on each draw) and
once in a full ``gc.collect()`` after saving.  Every session and the
healthcheck freeze while that runs.

Rendering now happens on a worker thread with a matplotlib ``Figure`` that
never touches pyplot's global state, so even the remaining ~1 s of drawing
no longer blocks other sessions.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import re
import threading
from io import BytesIO

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure

from pixelmap import backend
from pixelmap.gui import gui as gui_module
from tests.gui_helpers import select_preset

PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]
N_SHANKS = {"1.0": 1, "2.0-1shank": 1, "2.0-4shanks": 4, "QuadBase": 4}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _layout_args(gui):
    gui.update_filename()
    gui.generate_imro()
    return (gui.probe_type, gui.imro_list, gui.positions_file, gui.wiring_file, "t")


def _plot(gui, **kwargs):
    backend.plot_probe_layout(*_layout_args(gui), **kwargs)
    return plt.gcf().axes[0]


def _survey_kwargs(n_shanks):
    survey = {(s, e): "#%02x00ff" % ((e * 7) % 256) for s in range(n_shanks) for e in range(0, 1300, 3)}
    return {"survey_colors": survey, "survey_range": (0.0, 1.0)}


def _anatomy_kwargs(n_shanks):
    bands = [{"shank_id": s, "y_min": 200, "y_max": 900, "color": "#33aa55", "acronym": "CA1"} for s in range(n_shanks)]
    return {"anatomy_bands": bands}


def _strip_creation_date(pdf: bytes) -> bytes:
    return re.sub(rb"/CreationDate \(D:[^)]*\)", b"", pdf)


def _download(gui):
    return asyncio.run(gui.generate_pdf_content())


# --------------------------------------------------------------------------
# The download path (GUI)
# --------------------------------------------------------------------------

def test_pdf_download_does_not_force_gc(gui, monkeypatch):
    select_preset(gui, "2.0-4shanks")
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(a) or 0)

    buf = _download(gui)

    assert buf is not None and buf.getvalue().startswith(b"%PDF")
    assert calls == []


def test_pdf_download_not_ready_returns_none(gui, monkeypatch):
    monkeypatch.setattr(gui, "ready_to_download", lambda: False)
    assert _download(gui) is None


def test_pdf_download_leaves_no_pyplot_figures(gui):
    select_preset(gui, "2.0-4shanks")
    plt.close("all")
    _download(gui)
    assert plt.get_fignums() == []


def test_pdf_renders_off_the_event_loop_thread(gui, monkeypatch):
    select_preset(gui, "1.0")
    real = backend.probe_layout_pdf
    threads = []

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(backend, "probe_layout_pdf", spy)
    buf = _download(gui)

    assert buf.getvalue().startswith(b"%PDF")
    assert threads and threads[0] is not threading.current_thread()


def test_event_loop_runs_while_pdf_renders(gui, monkeypatch):
    """Other sessions' callbacks must run while a render is in progress."""
    select_preset(gui, "1.0")
    started, release = threading.Event(), threading.Event()

    def slow_render(*args, **kwargs):
        started.set()
        assert release.wait(10), "event loop never got to run"
        return BytesIO(b"%PDF-stub")

    monkeypatch.setattr(backend, "probe_layout_pdf", slow_render)

    async def main():
        download = asyncio.create_task(gui.generate_pdf_content())
        while not started.is_set():
            await asyncio.sleep(0.01)
        # The render is blocked; if it held the loop, this would never run.
        await asyncio.sleep(0)
        release.set()
        return await download

    assert asyncio.run(main()).getvalue() == b"%PDF-stub"


def test_concurrent_pdf_renders_are_capped(gui, monkeypatch):
    """A burst of downloads queues instead of saturating every core."""
    select_preset(gui, "1.0")
    lock = threading.Lock()
    running, peak = [0], [0]

    def counting_render(*args, **kwargs):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        threading.Event().wait(0.05)
        with lock:
            running[0] -= 1
        return BytesIO(b"%PDF-stub")

    monkeypatch.setattr(backend, "probe_layout_pdf", counting_render)

    async def main():
        return await asyncio.gather(*(gui.generate_pdf_content() for _ in range(8)))

    results = asyncio.run(main())
    assert len(results) == 8
    assert 1 <= peak[0] <= gui_module.PDF_RENDER_WORKERS


# --------------------------------------------------------------------------
# pyplot-free rendering (backend)
# --------------------------------------------------------------------------

class _NoPyplot:
    def __getattr__(self, name):
        raise AssertionError(f"pyplot.{name} used on the pyplot-free path")


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
@pytest.mark.parametrize("extras", ["plain", "survey", "anatomy"])
def test_draw_probe_layout_never_touches_pyplot(gui, monkeypatch, probe_type, extras):
    select_preset(gui, probe_type)
    n_shanks = N_SHANKS[probe_type]
    kwargs = {"plain": {}, "survey": _survey_kwargs(n_shanks), "anatomy": _anatomy_kwargs(n_shanks)}[extras]
    args = _layout_args(gui)
    monkeypatch.setattr(backend, "plt", _NoPyplot())

    fig = Figure()
    assert backend.draw_probe_layout(fig, *args, **kwargs) is fig

    assert len(fig.axes) == (2 if extras == "survey" else 1)  # + colorbar axes
    assert plt.get_fignums() == []


@pytest.mark.parametrize("probe_type", ["1.0", "2.0-4shanks"])  # one single-shank, one 4-shank
def test_pyplot_free_pdf_matches_pyplot_pdf(gui, probe_type):
    """Same bytes as the pyplot path, so the refactor cannot change output."""
    select_preset(gui, probe_type)
    n_shanks = N_SHANKS[probe_type]
    args, kwargs = _layout_args(gui), _survey_kwargs(n_shanks)

    via_pyplot = BytesIO()
    backend.plot_probe_layout(*args, **kwargs)
    plt.savefig(via_pyplot, format="pdf", dpi=300, bbox_inches="tight")
    via_figure = backend.probe_layout_pdf(*args, **kwargs)

    assert _strip_creation_date(via_figure.getvalue()) == _strip_creation_date(via_pyplot.getvalue())


def test_concurrent_pyplot_free_renders_are_identical(gui):
    """Figures on separate threads must not bleed into each other."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for probe_type in ("1.0", "2.0-4shanks"):
        select_preset(gui, probe_type)
        n_shanks = N_SHANKS[probe_type]
        jobs.append((tuple(_layout_args(gui)), _survey_kwargs(n_shanks)))

    def render(job):
        args, kwargs = job
        return _strip_creation_date(backend.probe_layout_pdf(*args, **kwargs).getvalue())

    expected = [render(job) for job in jobs]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(render, jobs * 2))
    assert got == expected * 2


def test_plot_probe_layout_still_uses_pyplot(gui):
    """The notebook relies on the figure being pyplot-managed (inline display),
    and on a ``None`` return so the cell does not display it a second time."""
    select_preset(gui, "1.0")
    plt.close("all")
    assert backend.plot_probe_layout(*_layout_args(gui)) is None
    assert len(plt.get_fignums()) == 1
    assert len(plt.gcf().axes) == 1


def test_plot_probe_layout_save_plot(gui, tmp_path):
    select_preset(gui, "1.0")
    backend.plot_probe_layout(*_layout_args(gui), save_plot=True, saveDir=tmp_path)
    (saved,) = tmp_path.glob("*.pdf")
    assert saved.read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------------
# Rendering speed / output (backend)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_legend_location_is_pinned(gui, probe_type):
    """loc="best" re-runs an O(n_artists) search on every draw; it must be fixed."""
    select_preset(gui, probe_type)
    legend = _plot(gui).get_legend()
    assert legend is not None
    assert legend._loc != 0  # 0 == "best"


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_legend_matches_previous_best_position(gui, probe_type):
    """The pinned spot is where loc="best" used to put it (output unchanged)."""
    select_preset(gui, probe_type)
    ax = _plot(gui)
    fig = ax.figure
    legend = ax.get_legend()
    fig.canvas.draw()
    pinned = legend.get_window_extent().bounds

    legend.set_loc("best")
    fig.canvas.draw()
    assert legend.get_window_extent().bounds == pytest.approx(pinned)


def _rgba(c):
    return tuple(round(float(v), 6) for v in c)


SELECTED_RGBA = _rgba(matplotlib.colors.to_rgba("red", 1.0))
UNSELECTED_RGBA = _rgba(matplotlib.colors.to_rgba("lightgrey", 0.7))
FORBIDDEN_RGBA = _rgba(matplotlib.colors.to_rgba("black", 1.0))
SURVEY_BAR_ZORDER = 50


def _electrode_collections(ax):
    """Per-shank electrode PatchCollections, in shank order (survey bars excluded)."""
    return [
        c
        for c in ax.collections
        if isinstance(c, PatchCollection) and c.get_zorder() != SURVEY_BAR_ZORDER
    ]


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
@pytest.mark.parametrize("with_survey", [False, True])
def test_electrodes_drawn_as_collections(gui, probe_type, with_survey):
    """Thousands of individual patches made each PDF take seconds to build/draw."""
    select_preset(gui, probe_type)
    n_shanks = N_SHANKS[probe_type]
    kwargs = {}
    if with_survey:
        # Includes an electrode with no survey value -> falls back to grey.
        kwargs = {"survey_colors": {(0, 0): "#ff0000"}, "survey_range": (0.0, 1.0)}
    ax = _plot(gui, **kwargs)

    collections = [c for c in ax.collections if isinstance(c, PatchCollection)]
    per_shank = 2 if with_survey else 1
    assert len(collections) == n_shanks * per_shank
    # Only shank outlines / anatomy bands remain as standalone patches.
    assert len(ax.patches) < 50
    # Every electrode is still drawn, exactly once.
    drawn = sum(len(c.get_paths()) for c in _electrode_collections(ax))
    assert drawn == len(gui.positions_df)


def test_selected_electrodes_coloured(gui):
    """Selected electrodes keep the opaque red facecolour inside the collection."""
    select_preset(gui, "2.0-1shank")
    (electrodes,) = _electrode_collections(_plot(gui))

    colours = [_rgba(c) for c in electrodes.get_facecolor()]
    assert set(colours) <= {SELECTED_RGBA, UNSELECTED_RGBA, FORBIDDEN_RGBA}
    assert colours.count(SELECTED_RGBA) == len(gui.imro_list) - 1  # minus header row


@pytest.mark.parametrize("probe_type", ["2.0-4shanks", "QuadBase"])
def test_multishank_forbidden_matches_shank_electrode_pairs(gui, probe_type, monkeypatch):
    """An electrode is black only if its (shank, electrode) pair is forbidden.

    Shank and electrode IDs used to be matched independently, so an electrode ID
    forbidden on one shank was also drawn black on every other shank that had
    any forbidden electrode.
    """
    select_preset(gui, probe_type)
    selected = np.array([[0, 0], [1, 1]])
    monkeypatch.setattr(backend, "find_selected_electrodes", lambda _imro: selected)
    forbidden = backend.find_forbidden_electrodes(selected, gui.wiring_df)
    forbidden_pairs = set(map(tuple, forbidden.tolist())) - set(map(tuple, selected.tolist()))

    ax = _plot(gui)
    positions = gui.positions_df.values
    drawn_selected, drawn_forbidden = set(), set()
    for shank_id, coll in enumerate(_electrode_collections(ax)):
        shank_positions = positions[positions[:, 0] == shank_id]
        colours = [_rgba(c) for c in coll.get_facecolor()]
        assert len(colours) == len(shank_positions)
        for (s, e, *_), colour in zip(shank_positions, colours):
            if colour == SELECTED_RGBA:
                drawn_selected.add((int(s), int(e)))
            elif colour == FORBIDDEN_RGBA:
                drawn_forbidden.add((int(s), int(e)))

    assert drawn_selected == {(0, 0), (1, 1)}
    assert drawn_forbidden == forbidden_pairs


@pytest.mark.parametrize("probe_type", ["2.0-1shank", "2.0-4shanks"])
def test_no_forbidden_electrodes(gui, probe_type, monkeypatch):
    """With no wiring conflicts, both branches draw without any black electrode."""
    select_preset(gui, probe_type)
    monkeypatch.setattr(
        backend, "find_forbidden_electrodes", lambda *_a: np.empty((0, 2), dtype=int)
    )

    ax = _plot(gui)

    colours = {_rgba(c) for coll in _electrode_collections(ax) for c in coll.get_facecolor()}
    assert FORBIDDEN_RGBA not in colours


def test_add_rects_empty_is_noop():
    _fig, ax = plt.subplots()
    backend._add_rects(ax, [], [])
    assert len(ax.collections) == 0


@pytest.mark.parametrize(
    "title, expected",
    [
        ("plain", "plain.pdf"),
        ("a_b c", "a_b_c.pdf"),
        ("two\nlines", "two_lines.pdf"),
        ("double__under", "double__under.pdf"),
    ],
)
def test_plot_probe_layout_save_filename(gui, tmp_path, title, expected):
    """Filenames match the pre-refactor rule: '_' and '\\n' become spaces, then
    spaces become '_'."""
    select_preset(gui, "1.0")
    args = _layout_args(gui)[:-1] + (title,)
    backend.plot_probe_layout(*args, save_plot=True, saveDir=tmp_path)
    assert [p.name for p in tmp_path.glob("*.pdf")] == [expected]


# --------------------------------------------------------------------------
# The wrappers forward drawing options to draw_probe_layout
# --------------------------------------------------------------------------

LEADING = ["probe_type", "imro_list", "positions_file", "wiring_file", "title"]
#: The real signature, taken before any test swaps the function out.
DRAW_PARAMS = inspect.signature(backend.draw_probe_layout).parameters


def _drawing_options():
    """draw_probe_layout's keyword options (everything after fig + LEADING)."""
    params = list(DRAW_PARAMS)
    assert params[:6] == ["fig", *LEADING]
    return params[6:]


@pytest.fixture
def recorded_draw(monkeypatch):
    """Replace draw_probe_layout with a recorder that also takes an option
    the wrappers have never heard of, standing in for future options."""
    calls = []

    def draw(fig, probe_type, imro_list, positions_file, wiring_file, title, *,
             brand_new_option=None, **options):
        calls.append(((probe_type, imro_list, positions_file, wiring_file, title),
                      dict(options, brand_new_option=brand_new_option)))
        fig.add_subplot()
        return fig

    monkeypatch.setattr(backend, "draw_probe_layout", draw)
    return calls


@pytest.mark.parametrize("wrapper", ["plot_probe_layout", "probe_layout_pdf"])
def test_wrappers_forward_every_drawing_option(recorded_draw, wrapper):
    options = {name: object() for name in _drawing_options()}
    options["brand_new_option"] = object()
    leading = tuple(object() for _ in LEADING)

    getattr(backend, wrapper)(*leading, **options)

    ((got_leading, got_options),) = recorded_draw
    assert got_leading == leading
    assert got_options == options


@pytest.mark.parametrize("wrapper", ["plot_probe_layout", "probe_layout_pdf"])
def test_wrappers_default_to_draw_probe_layouts_defaults(recorded_draw, wrapper):
    """Options left out are left to draw_probe_layout (figsize, which
    plot_probe_layout keeps as a positional, passes its matching default)."""
    getattr(backend, wrapper)(*LEADING)
    ((_, got_options),) = recorded_draw
    got_options.pop("brand_new_option")
    assert got_options == {name: DRAW_PARAMS[name].default for name in got_options}


def test_plot_probe_layout_keeps_its_positional_order():
    """Existing positional calls -- (..., title, figsize, save_plot, saveDir) --
    must still bind as before."""
    params = list(inspect.signature(backend.plot_probe_layout).parameters.values())
    assert [p.name for p in params[:8]] == [*LEADING, "figsize", "save_plot", "saveDir"]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params[:8])
    assert params[5].default == DRAW_PARAMS["figsize"].default
    assert params[6].default is False and params[7].default is None
    assert params[-1].kind is params[-1].VAR_KEYWORD


def test_plot_probe_layout_positional_save_args(gui, tmp_path, recorded_draw):
    select_preset(gui, "1.0")
    backend.plot_probe_layout(*_layout_args(gui), (3, 10), True, tmp_path)
    ((_, got_options),) = recorded_draw
    assert got_options["figsize"] == (3, 10)
    assert [p.name for p in tmp_path.glob("*.pdf")] == ["t.pdf"]


@pytest.mark.parametrize("wrapper", ["plot_probe_layout", "probe_layout_pdf"])
def test_wrapper_docstrings_point_to_draw_probe_layout(wrapper):
    assert ":func:`draw_probe_layout`" in getattr(backend, wrapper).__doc__
