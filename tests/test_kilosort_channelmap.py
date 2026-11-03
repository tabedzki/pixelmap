"""generate_kilosort_channelmap_dict: indexed position lookup, same output as the per-electrode scan."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap import backend, constants
from pixelmap.utils.imro import generate_imro_channelmap

WIRING_DIR = Path(backend.__file__).resolve().parent / "wiring_maps"

PRESETS = [
    (probe_type, preset)
    for probe_type, presets in [
        ("1.0", constants.SUPPORTED_1shank_PRESETS),
        ("2.0-1shank", constants.SUPPORTED_1shank_PRESETS),
        ("2.0-4shanks", constants.SUPPORTED_4shanks_PRESETS),
        ("QuadBase", constants.SUPPORTED_QuadBase_PRESETS),
    ]
    for preset in presets
]


def _files(probe_type):
    positions, wiring = constants.WIRING_FILE_MAP[probe_type]
    return WIRING_DIR / positions, WIRING_DIR / wiring


def _reference(imro_list, positions_df):
    """The per-electrode boolean-mask scan the index replaces."""
    selected = backend.find_selected_electrodes(imro_list)
    n = len(selected)
    xc, yc, kcoords = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    for i, (shank_id, electrode_id) in enumerate(selected):
        mask = (positions_df["shank"] == shank_id) & (positions_df["electrode"] == electrode_id)
        if not mask.any():
            raise ValueError(
                f"Electrode {electrode_id} on shank {shank_id} not found in positions file"
            )
        row = positions_df[mask].iloc[0]
        xc[i], yc[i], kcoords[i] = row["x"], row["y"], shank_id
    return {"chanMap": np.arange(n), "xc": xc, "yc": yc, "kcoords": kcoords, "n_chan": n}


def _assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual["n_chan"] == expected["n_chan"]
    assert type(actual["n_chan"]) is type(expected["n_chan"])
    for key in ("chanMap", "xc", "yc", "kcoords"):
        assert actual[key].dtype == expected[key].dtype, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _imro(probe_type, preset):
    return generate_imro_channelmap(probe_type, preset, wiring_file=_files(probe_type)[1])


@pytest.mark.parametrize(("probe_type", "preset"), PRESETS)
def test_matches_per_electrode_scan(probe_type, preset):
    positions_file, _ = _files(probe_type)
    imro_list = _imro(probe_type, preset)
    actual = backend.generate_kilosort_channelmap_dict(imro_list, positions_file)
    _assert_same(actual, _reference(imro_list, pd.read_csv(positions_file)))


def _with_selection(monkeypatch, selected):
    monkeypatch.setattr(
        backend, "find_selected_electrodes", lambda _imro: np.asarray(selected).reshape(-1, 2)
    )


def _csv(df):
    return io.StringIO(df.to_csv(index=False))


POSITIONS = pd.DataFrame(
    {"shank": [0, 0, 1, 1], "electrode": [0, 1, 0, 1], "x": [0, 32, 250, 282], "y": [0, 0, 15, 15]}
)


def test_empty_selection(monkeypatch):
    _with_selection(monkeypatch, [])
    actual = backend.generate_kilosort_channelmap_dict(None, _csv(POSITIONS))
    _assert_same(actual, _reference(None, POSITIONS))
    assert actual["n_chan"] == 0


def test_zero_ids_and_selection_order(monkeypatch):
    """Shank 0 / electrode 0 are real keys; output follows selection order, not file order."""
    _with_selection(monkeypatch, [[1, 1], [0, 0], [1, 0]])
    actual = backend.generate_kilosort_channelmap_dict(None, _csv(POSITIONS))
    np.testing.assert_array_equal(actual["xc"], [282, 0, 250])
    np.testing.assert_array_equal(actual["yc"], [15, 0, 15])
    np.testing.assert_array_equal(actual["kcoords"], [1, 0, 1])
    _assert_same(actual, _reference(None, POSITIONS))


def test_duplicate_positions_first_row_wins(monkeypatch):
    duplicated = pd.concat([POSITIONS, POSITIONS.assign(x=-1, y=-1)], ignore_index=True)
    _with_selection(monkeypatch, [[0, 1], [1, 0]])
    actual = backend.generate_kilosort_channelmap_dict(None, _csv(duplicated))
    np.testing.assert_array_equal(actual["xc"], [32, 250])
    _assert_same(actual, _reference(None, duplicated))


def test_columns_found_by_name_with_float_values(monkeypatch):
    shuffled = POSITIONS.assign(extra=7)[["y", "electrode", "extra", "x", "shank"]].astype(float)
    _with_selection(monkeypatch, [[0, 1], [1, 1]])
    actual = backend.generate_kilosort_channelmap_dict(None, _csv(shuffled))
    np.testing.assert_array_equal(actual["xc"], [32, 282])
    _assert_same(actual, _reference(None, shuffled))


@pytest.mark.parametrize("missing", [[2, 0], [0, 2], [0, -1]])
def test_missing_electrode_raises(monkeypatch, missing):
    _with_selection(monkeypatch, [[0, 0], missing, [5, 5]])
    with pytest.raises(
        ValueError, match=rf"Electrode {missing[1]} on shank {missing[0]} not found"
    ):
        backend.generate_kilosort_channelmap_dict(None, _csv(POSITIONS))
