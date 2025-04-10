from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import Samples
from boostedhh import utils
from boostedhh.utils import Channel

from bbtautau import bbtautau_vars

base_filters = [
    [
        ("('ak8FatJetPt', '0')", ">=", 250),
        ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
        ("('ak8FatJetPt', '1')", ">=", 200),
        # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
        # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
        # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
    ],
]


def trigger_filter(
    triggers: list[str], base_filters: list[tuple] = base_filters
) -> list[list[tuple]]:
    filters = [base_filters + [(f"('{trigger}', '0')", "==", 1)] for trigger in triggers]
    return filters


def get_columns(
    year: str,
    channel: Channel,
    triggers: bool = True,
    legacy_taggers: bool = True,
    ParT_taggers: bool = True,
):

    columns_data = [("weight", 1), ("ak8FatJetPt", 3)]
    if legacy_taggers:
        columns_data += [
            ("ak8FatJetPNetXbbLegacy", 3),
            ("ak8FatJetPNetQCDLegacy", 3),
            ("ak8FatJetPNetmassLegacy", 3),
            ("ak8FatJetParTmassResApplied", 3),
            ("ak8FatJetParTmassVisApplied", 3),
        ]

    columns_signal = copy.deepcopy(columns_data)

    if triggers:
        for branch in channel.triggers["data"][year]:
            columns_data.append((branch, 1))
        for branch in channel.triggers["signal"][year]:
            columns_signal.append((branch, 1))

    if ParT_taggers:
        for branch in [
            f"ak8FatJetParT{key}" for key in Samples.qcdouts + Samples.topouts + Samples.sigouts
        ]:
            columns_data.append((branch, 3))
            columns_signal.append((branch, 3))

    columns_signal += [
        ("GenTauhh", 1),
        ("GenTauhmu", 1),
        ("GenTauhe", 1),
    ]

    columns_bg = copy.deepcopy(columns_signal)  # for now

    columns = {"data": columns_data, "signal": columns_signal, "bg": columns_bg}
    return columns


def delete_columns(
    events_dict: dict[str, pd.DataFrame], channel: Channel, year: str, triggers=True
):
    for key, sample_events in events_dict.items():
        if triggers:
            sample_events.drop(
                columns=set(sample_events.columns)
                - set(channel.triggers[Samples.get_stype(key)][year]),
            )
    return


def load_samples(
    year: str,
    channel: Channel,
    paths: dict[str],
    filters: list[list[tuple]],
    load_columns: list[tuple] = None,
):
    events_dict = {}

    samples = Samples.SAMPLES.copy()

    # remove unnecessary data samples
    for key in Samples.DATASETS:
        if (key in samples) and (key not in channel.data_samples):
            del samples[key]

    # load only the specified columns
    if load_columns is not None:
        for key, sample in samples.items():
            sample.load_columns = load_columns[Samples.get_stype(key)]

    # load samples
    for key, sample in samples.items():
        if sample.selector is not None:
            events_dict[key] = utils.load_sample(sample, year, paths, filters)

    # keep only the specified bbtt channel
    for signal in Samples.SIGNALS:
        events_dict[f"{signal}{channel.key}"] = events_dict[signal][
            events_dict[signal][f"GenTau{channel.key}"][0]
        ]
        del events_dict[signal]

    return events_dict


def remove_overlap(events_dict: dict[str, pd.DataFrame], year: str, channel: Channel):
    # data overlap removal
    ldataset = channel.lepton_dataset

    trigdict = {"jetmet": {}, "tau": {}}
    if channel.isLepton:
        trigdict[ldataset] = {}
        lepton_triggers = utils.list_intersection(channel.lepton_triggers, channel.triggers[year])

    jet_triggers = utils.list_intersection(bbtautau_vars.HLT_jets, channel.triggers[year])
    tau_triggers = utils.list_intersection(bbtautau_vars.HLT_taus, channel.triggers[year])

    for key, d in trigdict.items():
        d["jets"] = np.sum([events_dict[key][hlt][0] for hlt in jet_triggers], axis=0).astype(bool)

        if key == "jetmet":
            continue

        d["taus"] = np.sum([events_dict[key][hlt][0] for hlt in tau_triggers], axis=0).astype(bool)
        d["taunojets"] = ~d["jets"] & d["taus"]

        if key == "tau":
            continue

        if channel.isLepton:
            d[ldataset] = np.sum(
                [events_dict[key][hlt][0] for hlt in lepton_triggers], axis=0
            ).astype(bool)

            d[f"{ldataset}noothers"] = ~d["jets"] & ~d["taus"] & d[ldataset]

    # remove overlap
    events_dict["jetmet"] = events_dict["jetmet"][trigdict["jetmet"]["jets"]]
    events_dict["tau"] = events_dict["tau"][trigdict["tau"]["taunojets"]]
    events_dict[ldataset] = events_dict[ldataset][trigdict[ldataset][f"{ldataset}noothers"]]

    return events_dict


def apply_triggers(events_dict: dict[str, pd.DataFrame], year: str, channel: Channel):
    """Apply triggers in MC and data, and remove overlap between datasets."""
    # MC
    for skey, events in events_dict.items():
        if skey not in Samples.DATASETS:
            triggered = np.sum([events[hlt][0] for hlt in channel.triggers], axis=0).astype(bool)
            events_dict[skey] = events[triggered]

    return remove_overlap(events_dict, year, channel)
