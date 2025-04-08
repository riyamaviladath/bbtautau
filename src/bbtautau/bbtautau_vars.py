"""Common variables for bbtautau analysis."""

from __future__ import annotations

import copy

years_2022 = ["2022", "2022EE"]
years_2023 = ["2023", "2023BPix"]
years = years_2022 + years_2023

cat = ["data", "MC"]

HLT_2022 = {
    "PNet": [
        "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
    ],
    "PFJet": [
        "HLT_AK8PFJet420_MassSD30",
        "HLT_AK8PFJet425_SoftDropMass40",
    ],
    # TODO: do we need QuadJet?
    "QuadJet": [
        "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
        "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
        "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
    ],
    "SingleTau": [
        "HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
    ],
    "DiTau": [
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
        # VBF DiTau
        # "HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
        # "HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1",
    ],
    "DitauJet": [
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
    ],
    "Muon": [
        "HLT_IsoMu24",
        # TODO: check sensitivity without below triggers
        "HLT_Mu50",
    ],
    # These are in Muon
    "MuonTau": [
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
    ],
    "EGamma": [
        "HLT_Ele30_WPTight_Gsf",
        # TODO: check sensitivity without below triggers
        "HLT_Ele115_CaloIdVT_GsfTrkIdT",
        "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        "HLT_Photon200",
    ],
    # This is in EGamma
    "ETau": [
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        # "HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1",  # Not sure what this is
    ],
    "MET": [
        "HLT_PFMET120_PFMHT120_IDTight",
    ],
}

HLT_2023 = {
    "PNet": [
        "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
        "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
        "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
    ],
    "PFJet": [
        "HLT_AK8PFJet420_MassSD30",
        "HLT_AK8PFJet425_SoftDropMass40",
    ],
    # TODO: do we need QuadJet after removing Parking?
    "QuadJet": [
        "HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65",  # absent from 2023 MC and all bpix
        "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
        "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
    ],
    "SingleTau": [
        "HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
    ],
    "DiTau": [
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
    ],
    "DitauJet": [
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
    ],
    "Muon": [
        "HLT_IsoMu24",
        "HLT_Mu50",
    ],
    "MuonTau": [
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
    ],
    "EGamma": [
        "HLT_Ele30_WPTight_Gsf",
        "HLT_Ele115_CaloIdVT_GsfTrkIdT",
        "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        "HLT_Photon200",
    ],
    "ETau": [
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
    ],
    "Parking": [
        "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55",
        "HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
    ],
    "MET": [
        "HLT_PFMET120_PFMHT120_IDTight",
    ],
}

HLT_dict = {c: {year: {} for year in years} for c in cat}
for c in cat:
    for y in years_2022:
        HLT_dict[c][y] = copy.deepcopy(HLT_2022)
    for y in years_2023:
        HLT_dict[c][y] = copy.deepcopy(HLT_2023)

# Manually remove some HLTs
HLT_dict["data"]["2022"]["MuonTau"].remove(
    "HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1"
)
HLT_dict["data"]["2022"]["ETau"].remove(
    "HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1"
)

# HLT_dict["MC"]["2022"]["ETau"].remove("HLT_IsoMu27_MediumDeepTauPFTauHPS20_eta2p1_SingleL1") is in skimmer but not in file

HLT_dict["data"]["2023"]["PNet"].remove("HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06")
HLT_dict["data"]["2023"]["PNet"].remove("HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03")
HLT_dict["data"]["2023"]["Parking"].remove("HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55")

HLT_dict["MC"]["2023"]["PNet"].remove("HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35")
HLT_dict["MC"]["2023"]["PNet"].remove("HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30")
HLT_dict["MC"]["2023"]["QuadJet"].remove("HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65")

HLT_dict["data"]["2023BPix"]["PNet"].remove("HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35")
HLT_dict["data"]["2023BPix"]["PNet"].remove(
    "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30"
)
HLT_dict["data"]["2023BPix"]["QuadJet"].remove("HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65")

HLT_dict["MC"]["2023BPix"]["PNet"].remove("HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35")
HLT_dict["MC"]["2023BPix"]["PNet"].remove(
    "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30"
)
HLT_dict["MC"]["2023BPix"]["QuadJet"].remove("HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65")

# combine into a single list
HLT_list = {
    c: {year: [hlt for sublist in HLT_dict[c][year].values() for hlt in sublist] for year in years}
    for c in cat
}
HLT_jets = {
    c: {
        year: [hlt for key in ["PNet", "PFJet","QuadJet"] for hlt in HLT_dict[c][year][key]] for year in years
    }
    for c in cat
    
}
HLT_taus = {
    year: [hlt for key in ["DiTau", "SingleTau","DiTauJet"] for hlt in HLT_dict[year][key]] for year in years
}
HLT_hh = {
    c: {
        year: [
            hlt
            for key in ["PNet", "PFJet", "QuadJet", "DiTau", "SingleTau"]
            for hlt in HLT_dict[c][year][key]
        ]
        for year in years
    }
    for c in cat
}
HLT_hmu = {
    c: {
        year: [
            hlt
            for key in ["PNet", "PFJet", "Muon", "MuonTau", "DiTau", "SingleTau"]
            for hlt in HLT_dict[c][year][key]
        ]
        for year in years
    }
    for c in cat
}
HLT_he = {
    c: {
        year: [
            hlt
            for key in ["PNet", "PFJet", "EGamma", "ETau", "DiTau", "SingleTau"]
            for hlt in HLT_dict[c][year][key]
        ]
        for year in years
    }
    for c in cat
}
