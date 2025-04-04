"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    MuonArray,
    TauArray,
)


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)
    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass
    fatjets["globalParT_QCD"] = (
        fatjets.globalParT_QCD0HF + fatjets.globalParT_QCD1HF + fatjets.globalParT_QCD2HF
    )
    fatjets["globalParT_XbbvsQCD"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT_{tautau}vsQCD"] = fatjets[f"globalParT_{tautau}"] / (
            fatjets[f"globalParT_{tautau}"] + fatjets["globalParT_QCD"]
        )

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
    object_pt: float,  # select objects based on this
    pt: float,  # make event selections based on this  # noqa: ARG001
    eta: float,
    msd: float,  # noqa: ARG001
    mreg: float,  # noqa: ARG001
    mreg_str="particleNet_mass_legacy",  # noqa: ARG001
):
    fatjet_sel = (
        fatjets.isTight
        & (fatjets.pt > object_pt)
        & (abs(fatjets.eta) < eta)
        # & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
    )
    return fatjets[fatjet_sel]


"""
Trigger quality bits in NanoAOD v12
0 => CaloIdL_TrackIdL_IsoVL,
1 => 1e (WPTight),
2 => 1e (WPLoose),
3 => OverlapFilter PFTau,
4 => 2e,
5 => 1e-1mu,
6 => 1e-1tau,
7 => 3e,
8 => 2e-1mu,
9 => 1e-2mu,
10 => 1e (32_L1DoubleEG_AND_L1SingleEGOr),
11 => 1e (CaloIdVT_GsfTrkIdT),
12 => 1e (PFJet),
13 => 1e (Photon175_OR_Photon200) for Electron;
"""


def good_electrons(events, electrons: ElectronArray):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    if events["HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1"]:
        ptcut = 25
    elif events["HLT_Ele30_WPTight_Gsf"]:
        ptcut = 31
    else:
        ptcut = 20

    ele_sel = (
        electrons.isTight
        & (electrons.pt > ptcut)
        & (abs(electrons.eta) < 2.5)
        & (abs(electrons.dz) < 0.2)
        & (abs(electrons.dxy) < 0.045)
    )

    return electrons[ele_sel]


"""
Trigger quality bits in NanoAOD v12
0 => TrkIsoVVL,
1 => Iso,
2 => OverlapFilter PFTau,
3 => 1mu,
4 => 2mu,
5 => 1mu-1e,
6 => 1mu-1tau,
7 => 3mu,
8 => 2mu-1e,
9 => 1mu-2e,
10 => 1mu (Mu50),
11 => 1mu (Mu100),
12 => 1mu-1photon for Muon;
"""


def good_muons(events, muons: MuonArray):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    if events["HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1"]:
        ptcut = 22
    elif events["HLT_IsoMu24"]:
        ptcut = 26
    else:
        ptcut = 20

    muon_sel = (
        muons.isTight
        & (muons.pt > ptcut)
        & (abs(muons.eta) < 2.5)
        & (abs(muons.dz) < 0.2)
        & (abs(muons.dxy) < 0.045)
    )

    return muons[muon_sel]


"""
Trigger quality bits in NanoAOD v12
0 => LooseChargedIso,
1 => MediumChargedIso,
2 => TightChargedIso,
3 => DeepTau,
4 => TightID OOSC photons,
5 => HPS,
6 => charged iso di-tau,
7 => deeptau di-tau,
8 => e-tau,
9 => mu-tau,
10 => single-tau/tau+MET,
11 => run 2 VBF+ditau,
12 => run 3 VBF+ditau,
13 => run 3 double PF jets + ditau,
14 => di-tau + PFJet,
15 => Displaced Tau,
16 => Monitoring,
17 => regional paths,
18 => L1 seeded paths,
19 => 1 prong tau paths for Tau;
"""


def good_taus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 32) & (abs(taus.eta) < 2.5) & (abs(taus.dz) < 0.2)
    return taus[tau_sel]


"""
Trigger quality bits in NanoAOD v12
0 => HLT_AK8PFJetX_SoftDropMass40_PFAK8ParticleNetTauTau0p30,
1 => hltAK8SinglePFJets230SoftDropMass40PNetTauTauTag0p03 for BoostedTau;
"""


def good_boostedtaus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 20) & (abs(taus.eta) < 2.5)
    return taus[tau_sel]
