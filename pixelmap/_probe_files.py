"""Locate the probe positions / wiring CSVs that ship with pixelmap."""

from pathlib import Path

from pixelmap.constants import WIRING_FILE_MAP

WIRING_MAPS_DIR = Path(__file__).resolve().parent / "wiring_maps"

_KINDS = ("positions", "wiring")


def packaged_probe_file(probe_type, kind):
    """Path of the packaged ``kind`` CSV (``"positions"`` or ``"wiring"``) for ``probe_type``.

    Used when a caller leaves ``positions_file`` / ``wiring_file`` as ``None``.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    try:
        files = WIRING_FILE_MAP[probe_type]
    except (KeyError, TypeError):
        raise ValueError(
            f"No packaged {kind} file for probe type {probe_type!r}; pass the file path explicitly "
            f"or use one of {list(WIRING_FILE_MAP)}"
        ) from None
    return WIRING_MAPS_DIR / files[_KINDS.index(kind)]
