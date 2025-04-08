from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import hh_vars, plotting, utils
from boostedhh.utils import PAD_VAL
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

from bbtautau import bbtautau_vars

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# parser = argparse.ArgumentParser()
# parser.add_argument("--channel", type=str, required=True) # "hadronic", "electron", "muon"
# parser.add_argument("--years", type=str, nargs="+", required=True)#"2023", "2023BPix", "2022","2022EE",
# args = parser.parse_args()

# CHANNEL = args.channel
# years = args.years

# Global variables
MAIN_DIR = Path("/home/users/lumori/bbtautau/")
SIG_KEYS = {"hadronic": "bbtthh", "electron": "bbtthe", "muon": "bbtthmu"}
ALL_TRIGGERS = {
    "hadronic": bbtautau_vars.HLT_hh,
    "muon": bbtautau_vars.HLT_hmu,
    "electron": bbtautau_vars.HLT_he,
}
LEPTON_TRIGGERS = {
    "hadronic": None,
    "electron": bbtautau_vars.HLT_he,
    "muon": bbtautau_vars.HLT_hmu,
}

tags = {
    "data": {
        "2022": "24Nov21ParTMass_v12_private_signal",
        "2022EE": "25Jan22AddYears_v12_private_signal",
        "2023": "25Mar7Signal_v12_private_signal",
        "2023BPix": "25Mar7Signal_v12_private_signal",
    },
    "signal": {
        "2022": "24Nov21ParTMass_v12_private_signal",
        "2022EE": "25Jan22AddYears_v12_private_signal",
        "2023": "25Mar7Signal_v12_private_signal",
        "2023BPix": "25Mar7Signal_v12_private_signal",
    },
}

base_dir = {
    "2022": Path("/ceph/cms/store/user/rkansal/bbtautau/skimmer/"),
    "2022EE": Path("/ceph/cms/store/user/rkansal/bbtautau/skimmer/"),
    "2023": Path("/ceph/cms/store/user/lumori/bbtautau/skimmer/"),
    "2023BPix": Path("/ceph/cms/store/user/lumori/bbtautau/skimmer/"),
}


qcdouts = ["QCD0HF", "QCD1HF", "QCD2HF"]
topouts = ["TopW", "TopbW", "TopbWev", "TopbWmv", "TopbWtauhv", "TopbWq", "TopbWqq"][:2]
sigouts = ["Xtauhtauh", "Xtauhtaue", "Xtauhtaum", "Xbb"]


class Analyser:
    def __init__(self, years, channel):
        self.channel = channel
        self.years = years
        self.plot_dir = MAIN_DIR / f"plots/SensitivityStudy/25Mar7{channel}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.sig_key = SIG_KEYS[channel]
        self.data_keys = {
            "hadronic": ["jetmet", "tau"],
            "electron": ["jetmet", "tau", "egamma"],
            "muon": ["jetmet", "tau", "muon"],
        }[channel]
        self.taukey = {"hadronic": "Xtauhtauh", "electron": "Xtauhtaue", "muon": "Xtauhtaum"}[
            channel
        ]
        self.lepton_dataset = {"hadronic": None, "electron": "egamma", "muon": "muon"}[channel]
        self.columns_data = {
            year: [
                ("weight", 1),
                ("ak8FatJetPt", 3),
                ("ak8FatJetPNetXbbLegacy", 3),
                ("ak8FatJetPNetQCDLegacy", 3),
                ("ak8FatJetPNetmassLegacy", 3),
                ("ak8FatJetParTmassResApplied", 3),
                ("ak8FatJetParTmassVisApplied", 3),
            ]
            for year in years
        }
        self.columns_signal = copy.deepcopy(self.columns_data)
        self.events_dict = {year: {} for year in years}
        for year in self.years:
            for branch in ALL_TRIGGERS[self.channel]["data"][year]:
                self.columns_data[year].append((branch, 1))
            for branch in ALL_TRIGGERS[self.channel]["MC"][year]:
                self.columns_signal[year].append((branch, 1))

            for branch in [f"ak8FatJetParT{key}" for key in qcdouts + topouts + sigouts]:
                self.columns_data[year].append((branch, 3))
                self.columns_signal[year].append((branch, 3))

            self.columns_signal[year] += [
                ("GenTauhh", 1),
                ("GenTauhmu", 1),
                ("GenTauhe", 1),
            ]

    def load_events(self, year, tight_filter=False):
        # define samples to load
        self.samples = {
            "jetmet": utils.Sample(
                path=base_dir[year] / tags["data"][year],
                selector="JetHT|JetMET",
                label="JetMET",
                isData=True,
                year=year,
                load_columns=utils.format_columns(self.columns_data[year]),
            ),
            "tau": utils.Sample(
                path=base_dir[year] / tags["data"][year],
                selector="Tau_Run",
                label="Tau",
                isData=True,
                year=year,
                load_columns=utils.format_columns(self.columns_data[year]),
            ),
            "muon": utils.Sample(
                path=base_dir[year] / tags["data"][year],
                selector="Muon_Run",
                label="Muon",
                isData=True,
                year=year,
                load_columns=utils.format_columns(self.columns_data[year]),
            ),
            "egamma": utils.Sample(
                path=base_dir[year] / tags["data"][year],
                selector="EGamma_Run",
                label="EGamma",
                isData=True,
                year=year,
                load_columns=utils.format_columns(self.columns_data[year]),
            ),
            "bbtt": utils.Sample(
                path=base_dir[year] / tags["signal"][year],
                selector=hh_vars.bbtt_sigs["bbtt"][year],
                label=r"HHbb$\tau\tau$",
                isData=False,
                year=year,
                load_columns=utils.format_columns(self.columns_signal[year]),
            ),
        }
        for key in ["jetmet", "tau", "egamma", "muon"]:
            if key not in self.data_keys:
                del self.samples[key]

        kin_filters = [
            ("('ak8FatJetPt', '0')", ">=", 250),  # 250
            ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
            ("('ak8FatJetPt', '1')", ">=", 200),
            # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
            # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
            # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
        ]
        # check 'or' and 'and' syntax
        filters_data = (
            [
                kin_filters + [(f"('{trigger}', '0')", "==", 1)]
                for trigger in (ALL_TRIGGERS[self.channel]["data"][year])
            ]
            if tight_filter
            else [kin_filters]
        )
        filters_MC = (
            [
                kin_filters + [(f"('{trigger}', '0')", "==", 1)]
                for trigger in (ALL_TRIGGERS[self.channel]["MC"][year])
            ]
            if tight_filter
            else [kin_filters]
        )
        # print(f"Filters: {filters}")
        # dictionary that will contain all information (from all samples)

        for key, sample in self.samples.items():
            self.events_dict[year][key] = utils.load_sample(
                sample, filters_data if key != "bbtt" else filters_MC
            )

        self.events_dict[year]["bbtthh"] = self.events_dict[year]["bbtt"][
            self.events_dict[year]["bbtt"]["GenTauhh"][0]
        ]
        self.events_dict[year]["bbtthmu"] = self.events_dict[year]["bbtt"][
            self.events_dict[year]["bbtt"]["GenTauhmu"][0]
        ]
        self.events_dict[year]["bbtthe"] = self.events_dict[year]["bbtt"][
            self.events_dict[year]["bbtt"]["GenTauhe"][0]
        ]
        del self.events_dict[year]["bbtt"]

        if (
            not tight_filter
        ):  # backup for testing, would eventually remove; need to apply trigger filters
            for skey in SIG_KEYS.values():
                triggered = np.sum(
                    [
                        self.events_dict[year][skey][hlt].iloc[:, 0]
                        for hlt in ALL_TRIGGERS[self.channel][year]
                    ],
                    axis=0,
                ).astype(bool)
                self.events_dict[year][skey] = self.events_dict[year][skey][triggered]

    def remove_duplicates(self, year):
        trigdict = {"jetmet": {}, "tau": {}}
        if self.lepton_dataset:
            trigdict[self.lepton_dataset] = {}

        for key, d in trigdict.items():
            d["all"] = np.sum(
                [
                    self.events_dict[year][key][hlt].iloc[:, 0]
                    for hlt in ALL_TRIGGERS[self.channel]["data"][year]
                ],
                axis=0,
            ).astype(bool)
            d["jets"] = np.sum(
                [
                    self.events_dict[year][key][hlt].iloc[:, 0]
                    for hlt in bbtautau_vars.HLT_jets["data"][year]
                ],
                axis=0,
            ).astype(bool)
            d["taus"] = np.sum(
                [
                    self.events_dict[year][key][hlt].iloc[:, 0]
                    for hlt in bbtautau_vars.HLT_taus["data"][year]
                ],
                axis=0,
            ).astype(bool)

            d["taunojets"] = ~d["jets"] & d["taus"]

            if self.lepton_dataset:
                d[self.lepton_dataset] = np.sum(
                    [
                        self.events_dict[year][key][hlt].iloc[:, 0]
                        for hlt in LEPTON_TRIGGERS[self.channel]["data"][year]
                    ],
                    axis=0,
                ).astype(bool)

                d[f"{self.lepton_dataset}noothers"] = (
                    ~d["jets"] & ~d["taus"] & d[self.lepton_dataset]
                )

        self.events_dict[year]["jetmet"] = self.events_dict[year]["jetmet"][
            trigdict["jetmet"]["jets"]
        ]
        self.events_dict[year]["tau"] = self.events_dict[year]["tau"][trigdict["tau"]["taunojets"]]
        if self.lepton_dataset:
            self.events_dict[year][self.lepton_dataset] = self.events_dict[year][
                self.lepton_dataset
            ][trigdict[self.lepton_dataset][f"{self.lepton_dataset}noothers"]]

    def delete_cols(self, year):
        # inefficient but works. could be cleaned
        t_to_drop = list(ALL_TRIGGERS[self.channel]["data"][year])
        for key in self.events_dict[year]:
            for t in self.events_dict[year][key]:
                if t in t_to_drop:
                    self.events_dict[year][key].drop(t, axis=1)

        # self.events_dict[year][key].drop(
        # [list(ALL_TRIGGERS[self.channel]["MC" if key in SIG_KEYS.values() else "data"][year])], axis=1
        # ) #somehow doesn't work

    def extract_year(self, year):
        self.load_events(year, tight_filter=True)
        self.remove_duplicates(year)
        self.delete_cols(year)
        print("done with year ", year)

    def build_tagger_dict(self):
        self.taggers_dict = {year: {} for year in self.years}
        for year in self.years:
            for key, events in self.events_dict[year].items():
                tvars = {}

                tvars["PQCD"] = sum([events[f"ak8FatJetParT{k}"] for k in qcdouts]).to_numpy()
                tvars["PTop"] = sum([events[f"ak8FatJetParT{k}"] for k in topouts]).to_numpy()

                for disc in ["Xbb", self.taukey]:
                    tvars[f"{disc}vsQCD"] = np.nan_to_num(
                        events[f"ak8FatJetParT{disc}"]
                        / (events[f"ak8FatJetParT{disc}"] + tvars["PQCD"]),
                        nan=PAD_VAL,
                    )
                    tvars[f"{disc}vsQCDTop"] = np.nan_to_num(
                        events[f"ak8FatJetParT{disc}"]
                        / (events[f"ak8FatJetParT{disc}"] + tvars["PQCD"] + tvars["PTop"]),
                        nan=PAD_VAL,
                    )

                    # make sure not to choose padded jets below by accident
                    nojet3 = events["ak8FatJetPt"][2] == PAD_VAL
                    tvars[f"{disc}vsQCD"][:, 2][nojet3] = PAD_VAL
                    tvars[f"{disc}vsQCDTop"][:, 2][nojet3] = PAD_VAL

                tvars["PNetXbbvsQCD"] = np.nan_to_num(
                    events["ak8FatJetPNetXbbLegacy"]
                    / (events["ak8FatJetPNetXbbLegacy"] + events["ak8FatJetPNetQCDLegacy"]),
                    nan=PAD_VAL,
                )

                # jet assignment
                fjbbpick = np.argmax(tvars["XbbvsQCD"], axis=1)
                fjttpick = np.argmax(tvars[f"{self.taukey}vsQCD"], axis=1)
                overlap = fjbbpick == fjttpick
                fjbbpick[overlap] = np.argsort(tvars["XbbvsQCD"][overlap], axis=1)[:, 1]

                # convert ids to boolean masks
                fjbbpick_mask = np.zeros_like(tvars["XbbvsQCD"], dtype=bool)
                fjbbpick_mask[np.arange(len(fjbbpick)), fjbbpick] = True
                fjttpick_mask = np.zeros_like(tvars[f"{self.taukey}vsQCD"], dtype=bool)
                fjttpick_mask[np.arange(len(fjttpick)), fjttpick] = True

                tvars["bb_mask"] = fjbbpick_mask
                tvars["tautau_mask"] = fjttpick_mask
                self.taggers_dict[year][key] = tvars

    @staticmethod
    def get_jet_vals(vals, mask):
        # check if vals is a numpy array
        if not isinstance(vals, np.ndarray):
            vals = vals.to_numpy()
        return vals[mask]

    def compute_rocs(self, years, jets=None, discs=None):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")
        if jets is None:
            jets = ["bb", "tautau"]
        if discs is None:
            discs = [
                "XbbvsQCD",
                "XbbvsQCDTop",
                f"{self.taukey}vsQCD",
                f"{self.taukey}vsQCDTop",
                "PNetXbbvsQCD",
            ]
        if not hasattr(self, "rocs"):
            self.rocs = {}
        self.rocs["_".join(years)] = {jet: {} for jet in jets}
        for jet in jets:
            for i, disc in enumerate(discs):
                bg_scores = np.concatenate(
                    [
                        self.get_jet_vals(
                            self.taggers_dict[year][key][disc],
                            self.taggers_dict[year][key][f"{jet}_mask"],
                        )
                        for key in self.data_keys
                        for year in years
                    ]
                )
                bg_weights = np.concatenate(
                    [
                        self.events_dict[year][key]["finalWeight"]
                        for key in self.data_keys
                        for year in years
                    ]
                )

                sig_scores = np.concatenate(
                    [
                        self.get_jet_vals(
                            self.taggers_dict[year][self.sig_key][disc],
                            self.taggers_dict[year][self.sig_key][f"{jet}_mask"],
                        )
                        for year in years
                    ]
                )
                sig_weights = np.concatenate(
                    [self.events_dict[year][self.sig_key]["finalWeight"] for year in years]
                )

                fpr, tpr, thresholds = roc_curve(
                    np.concatenate([np.zeros_like(bg_scores), np.ones_like(sig_scores)]),
                    np.concatenate([bg_scores, sig_scores]),
                    sample_weight=np.concatenate([bg_weights, sig_weights]),
                )

                self.rocs["_".join(years)][jet][disc] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds,
                    "label": disc,
                    "color": plt.cm.tab10.colors[i],
                }

        return fpr, tpr, thresholds

    def plot_rocs(self, years):
        if not hasattr(self, "rocs") or "_".join(years) not in self.rocs:
            print(f"No ROC curves computed yet in years {years}")
        for jet, title in zip(
            ["bb", "tautau"], ["bb FatJet", rf"$\tau_h\tau_{self.channel[0]}$ FatJet"]
        ):
            plotting.multiROCCurveGrey(
                {"": self.rocs["_".join(years)][jet]},
                title=title + "+".join(years),
                show=True,
                plot_dir=self.plot_dir,
                name=f"roc_{jet+'_'.join(years)}",
            )

    # here could block and save only data that needs after

    def prepare_sensitivity(self, years):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")

        mbbk = "ParTmassResApplied"
        mttk = "ParTmassResApplied"

        self.txbbs = {year: {} for year in years}
        self.txtts = {year: {} for year in years}
        self.masstt = {year: {} for year in years}
        self.massbb = {year: {} for year in years}
        self.ptbb = {year: {} for year in years}

        # precompute to speedup
        for year in years:
            for key in [self.sig_key] + self.data_keys:
                self.txbbs[year][key] = self.get_jet_vals(
                    self.taggers_dict[year][key]["XbbvsQCD"],
                    self.taggers_dict[year][key]["bb_mask"],
                )
                self.txtts[year][key] = self.get_jet_vals(
                    self.taggers_dict[year][key][f"{self.taukey}vsQCDTop"],
                    self.taggers_dict[year][key]["tautau_mask"],
                )
                self.masstt[year][key] = self.get_jet_vals(
                    self.events_dict[year][key][f"ak8FatJet{mttk}"],
                    self.taggers_dict[year][key]["tautau_mask"],
                )
                self.massbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key][f"ak8FatJet{mbbk}"],
                    self.taggers_dict[year][key]["bb_mask"],
                )
                self.ptbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key]["ak8FatJetPt"],
                    self.taggers_dict[year][key]["bb_mask"],
                )

    def compute_sig_bg(self, years, txbbcut, txttcut, mbb1, mbb2, mbbw2, mtt1, mtt2):
        bg_yield = 0
        sig_yield = 0
        for year in years:
            for key in [self.sig_key] + self.data_keys:
                if key == self.sig_key:
                    cut = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    sig_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut])
                else:
                    cut = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.ptbb[year][key] > 250)
                    )
                    msb1 = (self.massbb[year][key] > (mbb1 - mbbw2)) & (
                        self.massbb[year][key] < mbb1
                    )
                    msb2 = (self.massbb[year][key] > mbb2) & (
                        self.massbb[year][key] < (mbb2 + mbbw2)
                    )
                    bg_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut & msb1])
                    bg_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut & msb2])
        return sig_yield, bg_yield

    def sig_bkg_opt(
        self, years, gridsize=10, gridlims=(0.7, 1), B=1, normalize_sig=True, plot=False
    ):
        """
        Will have to improve definition of global params
        """

        # bbeff, tteff = 0.44,0.36 #0.44, 0.36 values determined by highest sig for 1 bkg event
        mbb1, mbb2 = 110.0, 160.0
        mbbw2 = (mbb2 - mbb1) / 2
        mtt1, mtt2 = 50, 1500

        bbcut = np.linspace(*gridlims, gridsize)
        ttcut = np.linspace(*gridlims, gridsize)

        BBcut, TTcut = np.meshgrid(bbcut, ttcut)

        # scalar function, must be vectorized
        def sig_bg(bbcut, ttcut):
            return self.compute_sig_bg(
                years=years,
                txbbcut=bbcut,
                txttcut=ttcut,
                mbb1=mbb1,
                mbb2=mbb2,
                mbbw2=mbbw2,
                mtt1=mtt1,
                mtt2=mtt2,
            )

        sigs, bgs = np.vectorize(sig_bg)(BBcut, TTcut)
        if normalize_sig:
            tot_sig_weight = np.sum(
                np.concatenate(
                    [self.events_dict[year][self.sig_key]["finalWeight"] for year in years]
                )
            )
            sigs = sigs / tot_sig_weight

        sel = (bgs >= 1) & (bgs <= B)
        if np.sum(sel) == 0:
            B_initial = B
            while np.sum(sel) == 0 and B < 100:
                B += 1
                sel = (bgs >= 1) & (bgs <= B)
            print(
                f"Need a finer grid, no region with B={B_initial}. I'm extending the region to B in [1,{B}].",
                bgs,
            )
        sel_idcs = np.argwhere(sel)
        opt_i = np.argmax(sigs[sel])
        max_sig_idx = tuple(sel_idcs[opt_i])
        bbcut_opt, ttcut_opt = BBcut[max_sig_idx], TTcut[max_sig_idx]

        significance = np.where(bgs > 0, sigs / np.sqrt(bgs), 0)
        max_significance_i = np.unravel_index(np.argmax(significance), significance.shape)
        bbcut_opt_significance, ttcut_opt_significance = (
            BBcut[max_significance_i],
            TTcut[max_significance_i],
        )

        """
        extract from roc data the efficiencies for the cuts:
        """
        # bbeff = rocs[year]["bb"]["XbbvsQCD"]["tpr"][

        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            hep.cms.label(
                ax=ax,
                label="Work in Progress",
                data=True,
                year="+".join(years),
                com="13.6",
                fontsize=13,
                lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
            )
            sigmap = ax.contourf(BBcut, TTcut, sigs, levels=10, cmap="viridis")
            ax.contour(BBcut, TTcut, sel, colors="r")
            proxy = Line2D([0], [0], color="r", label="B=1" if B == 1 else f"B in [1,{B}]")
            ax.scatter(bbcut_opt, ttcut_opt, color="r", label="Max. signal cut")
            # ax.scatter(bbcut_opt_significance, ttcut_opt_significance, color="b", label="Max. significance cut")
            ax.set_xlabel("Xbb vs QCD cut")
            ax.set_ylabel("Xtauhtauh vs QCD cut")
            cbar = plt.colorbar(sigmap, ax=ax)
            cbar.set_label("Signal efficiency" if normalize_sig else "Signal yield")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(proxy)
            ax.legend(handles=handles, loc="lower left")
            plt.savefig(
                self.plot_dir / f"sig_bkg_opt_{'_'.join(years)}_B={B}.pdf", bbox_inches="tight"
            )
            plt.savefig(
                self.plot_dir / f"sig_bkg_opt_{'_'.join(years)}_B={B}.png", bbox_inches="tight"
            )
            plt.show()

        return (
            [sigs[max_sig_idx], bgs[max_sig_idx]],
            [bbcut_opt, ttcut_opt],
            [sigs[max_significance_i], bgs[max_significance_i]],
            [bbcut_opt_significance, ttcut_opt_significance],
        )

    @staticmethod
    def print_nicely(sig_yield, bg_yield, years):
        print(
            f"""

            Yield study year(s) {years}:

            """
        )

        print("Sig yield", sig_yield)
        print("BG yield", bg_yield)
        print("limit", 2 * np.sqrt(bg_yield) / sig_yield)

        if "2023" not in years or "2023BPix" not in years:
            print(
                "limit scaled to 22-23 all channels",
                2
                * np.sqrt(bg_yield)
                / sig_yield
                / np.sqrt(
                    hh_vars.LUMI["2022-2023"] / np.sum([hh_vars.LUMI[year] for year in years]) * 3
                ),
            )
        print(
            "limit scaled to 22-24 all channels",
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt(
                (124000 + hh_vars.LUMI["2022-2023"])
                / np.sum([hh_vars.LUMI[year] for year in years])
                * 3
            ),
        )
        print(
            "limit scaled to Run 3 all channels",
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt((360000) / np.sum([hh_vars.LUMI[year] for year in years]) * 3),
        )
        return

    def as_df(self, sig_yield, bg_yield, years):
        limits = {}
        limits["Sig_Yield"] = sig_yield
        limits["BG_Yield"] = bg_yield
        limits["Limit"] = 2 * np.sqrt(bg_yield) / sig_yield

        if "2023" not in years and "2023BPix" not in years:
            limits["Limit_scaled_22_23"] = (
                2
                * np.sqrt(bg_yield)
                / sig_yield
                / np.sqrt(
                    hh_vars.LUMI["2022-2023"] / np.sum([hh_vars.LUMI[year] for year in years]) * 3
                )
            )
        else:
            limits["Limit_scaled_22_23"] = np.nan

        limits["Limit_scaled_22_24"] = (
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt(
                (124000 + hh_vars.LUMI["2022-2023"])
                / np.sum([hh_vars.LUMI[year] for year in years])
                * 3
            )
        )

        limits["Limit_scaled_Run3"] = (
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt((360000) / np.sum([hh_vars.LUMI[year] for year in years]) * 3)
        )

        df_out = pd.DataFrame([limits])
        return df_out


if __name__ == "__main__":

    years = ["2022", "2022EE", "2023", "2023BPix"]

    for c in [
        "electron",
        "hadronic",
        "muon",
    ]:
        print(f"Channel: {c}")
        analyser = Analyser(years, c)
        for year in years:
            analyser.extract_year(year)
            print(f"Loaded {year} events")

        analyser.build_tagger_dict()
        analyser.compute_rocs(years)
        analyser.plot_rocs(years)
        print("ROCs computed for channel ", c)
        analyser.prepare_sensitivity(years)

        results = {}
        for B in [1, 2, 8]:
            yields_B, cuts_B, yields_max_significance, cuts_max_significance = analyser.sig_bkg_opt(
                years, gridsize=30, B=B, plot=True
            )
            sig_yield, bkg_yield = yields_B
            sig_yield_max_sig, bkg_yield_max_sig = (
                yields_max_significance  # not very clean rn, can be improved but should be the same
            )
            results[f"B={B}"] = analyser.as_df(sig_yield, bkg_yield, years)
            print("done with B=", B)
        results["Max_significance"] = analyser.as_df(sig_yield_max_sig, bkg_yield_max_sig, years)
        results_df = pd.concat(results, axis=0)
        results_df.index = results_df.index.droplevel(1)
        print(c, "\n", results_df.T.to_markdown())
        results_df.T.to_csv(analyser.plot_dir / f"{'_'.join(years)}-results_fast.csv")
        del analyser
