"""Helpers shared by the GUI download tests."""

from __future__ import annotations


def select_preset(gui, probe_type):
    gui.probe_type = probe_type
    gui.preset = next(p for p in gui.param.preset.objects if p)
    gui.apply_preset()
