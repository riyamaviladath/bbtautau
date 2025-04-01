"""Common variables for bbtautau analysis."""

from __future__ import annotations

years_2022 = ["2022", "2022EE"]
years_2023 = ["2023", "2023BPix"]
years = years_2022 + years_2023

HLT_2022 = {
    "PNet": [
        "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
    ],
    "PFJet": [
        "HLT_AK8PFJet425_SoftDropMass40",
    ],
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
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
    ],
    "Muon": [
        "HLT_IsoMu24",
        "HLT_Mu50",
    ],
    "MuonTau": [
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1",
        "HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1",
    ],
    "EGamma": [
        "HLT_Ele30_WPTight_Gsf",
        "HLT_Ele115_CaloIdVT_GsfTrkIdT",
        "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        "HLT_Photon200",
    ],
    "ETau": [
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        "HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1",
    ],
}

HLT_2023 = {
    "PNet": [
        "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
        "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
        "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
    ],
    "PFJet": ["HLT_AK8PFJet425_SoftDropMass40", "HLT_AK8PFJet420_MassSD30"],
    "QuadJet": [
        "HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65",  # prob will be absent from MC
        "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
        "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
    ],
    "SingleTau": [
        "HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
    ],
    "DiTau": [
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
    ],
    "Muon": [
        "HLT_IsoMu24",
        "HLT_Mu50",
    ],
    "MuonTau": [
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1",
        "HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
        "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1",
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
}

HLT_dict = {}
HLT_dict.update(dict.fromkeys(years_2022, HLT_2022))
HLT_dict.update(dict.fromkeys(years_2023, HLT_2023))

# combine into a single list
HLT_list = {year: [hlt for sublist in HLT_dict[year].values() for hlt in sublist] for year in years}
HLT_jets = {
    year: [hlt for key in ["PNet", "PFJet", "QuadJet"] for hlt in HLT_dict[year][key]]
    for year in years
}
HLT_taus = {
    year: [hlt for key in ["DiTau", "SingleTau"] for hlt in HLT_dict[year][key]] for year in years
}
HLT_hh = {
    year: [
        hlt
        for key in ["PNet", "PFJet", "QuadJet", "DiTau", "SingleTau"]
        for hlt in HLT_dict[year][key]
    ]
    for year in years
}
HLT_hmu = {
    year: [
        hlt
        for key in ["PNet", "PFJet", "Muon", "MuonTau", "DiTau", "SingleTau"]
        for hlt in HLT_dict[year][key]
    ]
    for year in years
}
HLT_he = {
    year: [
        hlt
        for key in ["PNet", "PFJet", "EGamma", "ETau", "DiTau", "SingleTau"]
        for hlt in HLT_dict[year][key]
    ]
    for year in years
}

HLT_map = {"hadronic": HLT_hh, "muon": HLT_hmu, "electron": HLT_he}
