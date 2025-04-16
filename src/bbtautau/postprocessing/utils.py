"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

from __future__ import annotations

import hist
import pandas as pd
from boostedhh import utils
from boostedhh.hh_vars import data_key
from boostedhh.utils import ShapeVar
from hist import Hist

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing import Samples


def get_var(events: pd.DataFrame, feat: str):
    if feat in events:
        return events[feat].to_numpy().squeeze()
    elif feat.startswith(("bb", "tt")):
        raise NotImplementedError("bb tautau assignment not yet implemented!")
    elif utils.is_int(feat[-1]):
        return events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()


def singleVarHist(
    events_dict: dict[str, pd.DataFrame],
    shape_var: ShapeVar,
    channel: Channel,
    weight_key: str = "finalWeight",
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist(
        hist.axis.StrCategory(samples + [data_key], name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for sample in samples:
        events = events_dict[sample]
        if Samples.SAMPLES[sample].isData and var.endswith(("_up", "_down")):
            fill_var = "_".join(var.split("_")[:-2])  # remove _up/_down
        else:
            fill_var = var

        fill_data = {var: get_var(events, fill_var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        # if sf is not None and year is not None and sample == "ttbar" and apply_tt_sf:
        #     weight = weight   * tau32FittedSF_4(events) * ttbar_pTjjSF(year, events)

        if fill_data[var] is not None:
            h.fill(Sample=sample, **fill_data, weight=weight)

    data_hist = sum(h[skey, ...] for skey in channel.data_samples)
    h.view(flow=True)[utils.get_key_index(h, data_key)].value = data_hist.values(flow=True)
    h.view(flow=True)[utils.get_key_index(h, data_key)].variance = data_hist.variances(flow=True)

    if shape_var.blind_window is not None:
        utils.blindBins(h, shape_var.blind_window, data_key)

    return h
