"""
Defines all the analysis regions.
****Important****: Region names used in the analysis cannot have underscores because of a rhalphalib convention.
Author(s): Raghav Kansal
"""

from __future__ import annotations

from dataclasses import dataclass

from utils import CUT_MAX_VAL

from bbtautau.bbtautau_utils import Channel


@dataclass
class Region:
    cuts: dict = None
    label: str = None
    signal: bool = False  # is this a signal region?
    cutstr: str = None  # optional label for the region's cuts e.g. when scanning cuts


def get_selection_regions(channel: Channel):
    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": [250, CUT_MAX_VAL],
                "ttFatJetPt": [200, CUT_MAX_VAL],
                # "ttFatJetMass"  # TODO
                "bbFatJetParTXbbvsQCD": [channel.txbb_cut, CUT_MAX_VAL],
                f"ttFatJetParT{channel.tagger}vsQCD": [channel.txtt_cut, CUT_MAX_VAL],
            },
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": [250, CUT_MAX_VAL],
                "ttFatJetPt": [200, CUT_MAX_VAL],
                # "ttFatJetMass"  # TODO
                "bbFatJetParTXbbvsQCD": [-CUT_MAX_VAL, channel.txbb_cut],
                f"ttFatJetParT{channel.tagger}vsQCD": [-CUT_MAX_VAL, channel.txtt_cut],
            },
            signal=False,
            label="Fail",
        ),
    }

    return regions
