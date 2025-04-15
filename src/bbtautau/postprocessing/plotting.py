"""
Plotting functions for bbtautau.

Authors: Raghav Kansal, add your names
"""

from __future__ import annotations

from boostedhh import plotting
from hist import Hist
from Samples import SAMPLES

bg_order = ["Diboson", "HH", "HWW", "Hbb", "ST", "W+Jets", "Z+Jets", "TT", "QCD"]

sample_label_map = {s: SAMPLES[s].label for s in SAMPLES}


def ratioHistPlot(hists: Hist, year: str, sig_keys: list[str], bg_keys: list[str], **kwargs):
    plotting.ratioHistPlot(
        hists,
        year,
        sig_keys,
        bg_keys,
        bg_order=bg_order,
        sample_label_map=sample_label_map,
        **kwargs,
    )
