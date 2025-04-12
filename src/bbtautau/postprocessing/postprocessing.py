from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import Samples
from boostedhh import utils
from boostedhh.utils import Channel

from bbtautau import bbtautau_vars

base_filters = [
    ("('ak8FatJetPt', '0')", ">=", 250),
    ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
    ("('ak8FatJetPt', '1')", ">=", 200),
    # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
    # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
    # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
]


def trigger_filter(
    triggers: dict[str, list[str]],
    base_filters: list[tuple] = base_filters,
    fast_mode: bool = False,
) -> dict[str, dict[str, list[list[tuple]]]]:
    """
    creates a list of filters for each trigger in the list of triggers. It is granular to the usual {"data" / "signal" : {year : [triggers],...} structure: triggers = {"data": {"2022" : [...] , ...}, "signal": { [...]}.
    """
    if fast_mode:
        base_filters += [("('ak8FatJetPNetXbbLegacy', '0')", ">=", 0.95)]

    filters_dic = {}
    for dtype, years in triggers.items():
        filters_dic[dtype] = {}
        for year, trigger_list in years.items():
            filters_dic[dtype][year] = [
                base_filters + [(f"('{trigger}', '0')", "==", 1)] for trigger in trigger_list
            ]
    # print(f"\n\nTrigger filters_dic for data: {filters_dic['data']['2022'][0]}")
    return filters_dic


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
        ("GenTauhmu", 1),  # TODO this will need to be changed to GenTauhm
        ("GenTauhe", 1),
    ]

    columns_bg = copy.deepcopy(columns_signal)  # for now

    columns = {
        "data": utils.format_columns(columns_data),
        "signal": utils.format_columns(columns_signal),
        "bg": utils.format_columns(columns_bg),
    }
    return columns


def load_samples(
    year: str,
    channel: Channel,
    paths: dict[str],
    filters_dic: dict[str, dict[str, list[list[tuple]]]] = None,
    load_columns: dict[str, list[tuple]] = None,
    load_bgs: bool = False,
    load_just_bbtt: bool = False,
):
    events_dict = {}

    samples = Samples.SAMPLES.copy()
    signals = Samples.SIGNALS.copy()

    if load_just_bbtt:  # quite ad hoc but should become obsolete
        del samples["vbfbbtt-k2v0"]
        signals.remove("vbfbbtt-k2v0")

    # remove unnecessary data samples
    for key in Samples.DATASETS + (not load_bgs) * Samples.BGS:
        if (key in samples) and (key not in channel.data_samples):
            del samples[key]

    # load only the specified columns
    if load_columns is not None:
        for key, sample in samples.items():
            sample.load_columns = load_columns[Samples.get_stype(key)]

    # load samples
    for key, sample in samples.items():
        if sample.selector is not None:
            print(f"Loading {key}...")
            events_dict[key] = utils.load_sample(
                sample,
                year,
                paths,
                filters_dic[Samples.get_stype(key)][year] if filters_dic is not None else None,
            )

    # keep only the specified bbtt channel
    for signal in signals:
        events_dict[f"{signal}{channel.key}"] = events_dict[signal][
            events_dict[signal][f"GenTau{channel.key}" + "u" * (channel.key == "hm")][
                0
            ]  # quick fix due to old naming still in samples
        ]
        del events_dict[signal]

    return events_dict


def remove_overlap(events_dict: dict[str, pd.DataFrame], year: str, channel: Channel):
    # data overlap removal (never done in MC)
    ldataset = channel.lepton_dataset

    trigdict = {"jetmet": {}, "tau": {}}
    if channel.isLepton:
        trigdict[ldataset] = {}
        lepton_triggers = utils.list_intersection(
            channel.lepton_triggers, channel.triggers["data"][year]
        )

    jet_triggers = utils.list_intersection(
        bbtautau_vars.HLT_jets["data"][year], channel.triggers["data"][year]
    )
    tau_triggers = utils.list_intersection(
        bbtautau_vars.HLT_taus["data"][year], channel.triggers["data"][year]
    )

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

            events_dict[ldataset] = events_dict[ldataset][trigdict[ldataset][f"{ldataset}noothers"]]

    # remove overlap
    # print(trigdict["jetmet"])

    events_dict["jetmet"] = events_dict["jetmet"][trigdict["jetmet"]["jets"]]
    events_dict["tau"] = events_dict["tau"][trigdict["tau"]["taunojets"]]

    return events_dict


def apply_triggers(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    channel: Channel,
    remove_overlap_flag: bool = True,
):
    """Apply triggers in MC and data, and remove overlap between datasets."""
    # MC
    for skey, events in events_dict.items():
        if skey in Samples.SIGNALS_CHANNELS:
            triggered = np.sum(
                [events[hlt][0] for hlt in channel.triggers["signal"][year]], axis=0
            ).astype(bool)
            events_dict[skey] = events[triggered]

    return remove_overlap(events_dict, year, channel) if remove_overlap_flag else events_dict


def delete_columns(
    events_dict: dict[str, pd.DataFrame], year: str, channel: Channel, triggers=True
):
    for sample_key, sample_events in events_dict.items():
        print(sample_key, len(sample_events))
        if triggers:
            sample_events.drop(
                columns=list(
                    set(sample_events.columns)
                    - set(channel.triggers[Samples.get_stype(sample_key)][year])
                )
            )
    return events_dict
