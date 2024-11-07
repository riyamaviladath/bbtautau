"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
from coffea.nanoevents.methods.nanoaod import (
    FatJetArray,
)


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)
    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass
    fatjets["globalParT_massResCorr"] = fatjets.globalParT_massRes
    fatjets["globalParT_massVisCorr"] = fatjets.globalParT_massVis
    fatjets["globalParT_massResApplied"] = (
        fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
    )
    fatjets["globalParT_massVisApplied"] = (
        fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass
    )
    return fatjets


# ak8 jet definition
def good_ak8jets(
    fatjets: FatJetArray,
    pt: float,
    eta: float,
    msd: float,  # noqa: ARG001
    mreg: float,  # noqa: ARG001
    mreg_str="particleNet_mass_legacy",  # noqa: ARG001
):
    fatjet_sel = (
        fatjets.isTight
        & (fatjets.pt > pt)
        & (abs(fatjets.eta) < eta)
        # & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
    )
    return fatjets[fatjet_sel]
