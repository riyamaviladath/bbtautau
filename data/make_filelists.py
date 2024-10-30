from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"

# request for non-SM signal samples
# https://gitlab.cern.ch/cms-hig-mc/mc-hig-requests/-/issues/55

# https://cms-pdmv-prod.web.cern.ch/pmp/present?r=HIG-Run3Summer22wmLHEGS-00351


qcd_bins = [
    # "0to80",
    "15to30",  # unclear if these are needed
    "30to50",
    "50to80",
    "80to120",
    "120to170",
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000to1400",
    "1400to1800",
    "1800to2400",
    "2400to3200",
    "3200",
]

qcd_mu_bins = [
    "120to170",
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000",
]

qcd_ht_bins = [
    # "40to70",
    "70to100",
    "40to100",
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1200",
    "1200to1500",
    "1500to2000",
    "2000",
]


def get_v12v2_private():
    return {
        "2022": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "VBFHH": {
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022/HH4b/VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8/",
            },
            "Hbb": {
                "GluGluHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "ZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ggZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "WtoLNu-2Jets_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/VJets/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
            },
            "SingleTop": {
                "TbarBQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TBbarQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TWminustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TWminusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022/SingleTop/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2022C_single": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/JetMET/JetHT/JetHT_Run2022C/",
                ],
                "JetMET_Run2022C": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/JetMET/JetMET/JetMET_Run2022C/",
                "JetMET_Run2022D": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/JetMET/JetMET/JetMET_Run2022D/",
            },
            "Muon": {
                "Muon_Run2022C_single": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/Muon/SingleMuon/SingleMuon_Run2022C/",
                "Muon_Run2022C": [
                    "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/Muon/Muon/Muon_Run2022C/",
                ],
                "Muon_Run2022D": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/Muon/Muon/Muon_Run2022D/",
            },
            "EGamma": {
                "EGamma_Run2022C": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/EGamma/EGamma/EGamma_Run2022C/",
                "EGamma_Run2022D": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022/EGamma/EGamma/EGamma_Run2022D/",
            },
        },
        "2022EE": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "VBFHH": {
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2022EE/HH4b/VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8/",
            },
            "Hbb": {
                "GluGluHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "ZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ggZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022EE/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "WtoLNu-2Jets_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/VJets/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
            },
            "SingleTop": {
                "TbarBQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TBbarQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TWminustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TWminusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2022EE/SingleTop/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2022E": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/JetMET/JetMET/JetMET_Run2022E/",
                "JetMET_Run2022F": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/JetMET/JetMET/JetMET_Run2022F/",
                "JetMET_Run2022G": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/JetMET/JetMET/JetMET_Run2022G/",
            },
            "Muon": {
                "Muon_Run2022E": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/Muon/Muon/Muon_Run2022E/",
                "Muon_Run2022F": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/Muon/Muon/Muon_Run2022F/",
                "Muon_Run2022G": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/Muon/Muon/Muon_Run2022G/",
                ],
            },
            "EGamma": {
                "EGamma_Run2022E": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/EGamma/EGamma/EGamma_Run2022E/",
                "EGamma_Run2022F": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/EGamma/EGamma/EGamma_Run2022F/",
                "EGamma_Run2022G": "/store/group/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2022EE/EGamma/EGamma/EGamma_Run2022G/",
            },
        },
        "2023": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "VBFHH": {
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023/HH4b/VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96_TuneCP5_13p6TeV_madgraph-pythia8/",
            },
            "Hbb": {
                "GluGluHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "ZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ggZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "WtoLNu-2Jets_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/VJets/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
            },
            "SingleTop": {
                "TbarBQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/SingleTop/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TBbarQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/SingleTop/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TWminustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/SingleTop/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TWminusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/SingleTop/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023/SingleTop/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2023Cv1": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET0/JetMET_Run2023C_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET1/JetMET_Run2023C_1v1/",
                ],
                "JetMET_Run2023Cv2": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET0/JetMET_Run2023C_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET1/JetMET_Run2023C_1v2/",
                ],
                "JetMET_Run2023Cv3": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET0/JetMET_Run2023C_0v3/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET1/JetMET_Run2023C_1v3/",
                ],
                "JetMET_Run2023Cv4": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET0/JetMET_Run2023C_0v4/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/JetMET/JetMET1/JetMET_Run2023C_1v4/",
                ],
            },
            "Muon": {
                "Muon_Run2023Cv1": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon0/Muon_Run2023C_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon1/Muon_Run2023C_1v1/",
                ],
                "Muon_Run2023Cv2": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon0/Muon_Run2023C_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon1/Muon_Run2023C_1v2/",
                ],
                "Muon_Run2023Cv3": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon0/Muon_Run2023C_0v3/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon1/Muon_Run2023C_1v3/",
                ],
                "Muon_Run2023Cv4": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon0/Muon_Run2023C_0v4/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/Muon/Muon1/Muon_Run2023C_1v4/",
                ],
            },
            "EGamma": {
                "EGamma_Run2023Cv1": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma0/EGamma_Run2023C_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma1/EGamma_Run2023C_1v1/",
                ],
                "EGamma_Run2023Cv2": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma0/EGamma_Run2023C_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma1/EGamma_Run2023C_1v2/",
                ],
                "EGamma_Run2023Cv3": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma0/EGamma_Run2023C_0v3/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma1/EGamma_Run2023C_1v3/",
                ],
                "EGamma_Run2023Cv4": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma0/EGamma_Run2023C_0v4/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023/EGamma/EGamma1/EGamma_Run2023C_1v4/",
                ],
            },
        },
        "2023BPix": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "VBFHH": {
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94_TuneCP5_13p6TeV_madgraph-pythia8/",
                "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/cmantill/2023BPix/HH4b/VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96_TuneCP5_13p6TeV_madgraph-pythia8/",
            },
            "Hbb": {
                "GluGluHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "ZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2C_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ggZH_Hto2C_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2023BPix/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "WtoLNu-2Jets_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "WtoLNu-2Jets_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_0J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "DYto2L-2Jets_MLL-50_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Wto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_1J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
                "Zto2Q-2Jets_PTQQ-600_2J": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/VJets/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/",
            },
            "SingleTop": {
                "TbarBQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TBbarQ_t-channel_4FS": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/",
                "TWminustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TWminusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplusto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/",
                "TbarWplustoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/woodson/2023BPix/SingleTop/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2023D": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/JetMET/JetMET0/JetMET_Run2023D_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/JetMET/JetMET0/JetMET_Run2023D_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/JetMET/JetMET1/JetMET_Run2023D_1v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/JetMET/JetMET1/JetMET_Run2023D_1v2/",
                ],
            },
            "Muon": {
                "Muon_Run2023D": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/Muon/Muon0/Muon_Run2023D_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/Muon/Muon0/Muon_Run2023D_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/Muon/Muon1/Muon_Run2023D_1v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/Muon/Muon1/Muon_Run2023D_1v2/",
                ],
            },
            "EGamma": {
                "EGamma_Run2023D": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/EGamma/EGamma0/EGamma_Run2023D_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/EGamma/EGamma0/EGamma_Run2023D_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/EGamma/EGamma1/EGamma_Run2023D_1v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12_ParT/sixie/2023BPix/EGamma/EGamma1/EGamma_Run2023D_1v2/",
                ],
            },
        },
    }


def eos_rec_search(startdir, suffix, dirs):
    # print(f"EOS Recursive search in {startdir}.")
    eosbase = "root://cmseos.fnal.gov/"
    try:
        dirlook = (
            subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
            .decode("utf-8")
            .split("\n")[:-1]
        )
    except:
        print(f"No files found for {startdir}")
        return dirs

    donedirs = [[] for d in dirlook]
    for di, d in enumerate(dirlook):
        # print(f"Looking in {dirlook}.")
        if d.endswith(suffix):
            donedirs[di].append(startdir + "/" + d)
        elif d == "log":
            continue
        else:
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


def get_files(dataset, version):
    if "private" in version:
        files = eos_rec_search(dataset, ".root", [])
        return [f"root://cmseos.fnal.gov/{f}" for f in files]
    else:
        import requests
        from rucio_utils import get_dataset_files, get_proxy_path

        proxy = get_proxy_path()
        if "USER" in dataset:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/phys03/DBSReader/files?dataset={dataset}&detail=True"
        else:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/files?dataset={dataset}&detail=True"
        r = requests.get(
            link,
            cert=proxy,
            verify=False,
        )
        filesjson = r.json()
        files = []
        not_valid = []
        for fj in filesjson:
            if "is_file_valid" in fj:
                if fj["is_file_valid"] == 0:
                    # print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                    not_valid.append(fj["logical_file_name"])
                else:
                    files.append(fj["logical_file_name"])
            else:
                continue

        if "USER" in dataset:
            files_valid = [f"root://cmseos.fnal.gov/{f}" for f in files]
            return files_valid

        if len(files) == 0:
            print(f"Found 0 files for sample {dataset}!")
            return []

        # Now query rucio to get the concrete dataset passing the sites filtering options
        sites_cfg = {
            "whitelist_sites": [],
            "blacklist_sites": [
                "T2_FR_IPHC" "T2_US_MIT",
                "T2_US_Vanderbilt",
                "T2_UK_London_Brunel",
                "T2_UK_SGrid_RALPP",
                "T1_UK_RAL_Disk",
                "T2_PT_NCG_Lisbon",
            ],
            "regex_sites": None,
        }
        if version == "v12" or version == "v11":
            sites_cfg["whitelist_sites"] = ["T1_US_FNAL_Disk"]

        files_rucio, sites = get_dataset_files(dataset, **sites_cfg, output="first")

        # print(dataset, sites)

        # Get rid of invalid files
        files_valid = []
        for f in files_rucio:
            invalid = False
            for nf in not_valid:
                if nf in f:
                    invalid = True
                    break
            if not invalid:
                files_valid.append(f)

        return files_valid


# for version in ["v12"]:
# for version in ["v9", "v9_private", "v9_hh_private", "v11", "v11_private"]:
for version in ["v12v2_private"]:
    datasets = globals()[f"get_{version}"]()
    index = datasets.copy()
    for year, ydict in datasets.items():
        print(year)
        for sample, sdict in ydict.items():
            print(sample)
            for sname, dataset in sdict.items():
                if isinstance(dataset, list):
                    files = []
                    for d in dataset:
                        files.extend(get_files(d, version))
                    index[year][sample][sname] = files
                else:
                    index[year][sample][sname] = get_files(dataset, version)

    with Path(f"nanoindex_{version}.json").open("w") as f:
        json.dump(index, f, indent=4)
