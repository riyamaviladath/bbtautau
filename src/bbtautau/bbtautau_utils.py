"""Utilities for bbtautau package."""

from __future__ import annotations

from dataclasses import dataclass

from bbtautau.HLTs import HLTs


@dataclass
class Channel:
    """Channel."""

    key: str  # key in dictionaries etc.
    label: str  # label for plotting
    data_samples: list[str]  # datasets for this channel
    hlt_types: list[str]  # list of HLT types
    triggers: list[str] | dict[str, list[str]]  # list of triggers or dict of triggers per year
    isLepton: bool  # lepton channel or fully hadronic
    lepton_dataset: str = None  # lepton dataset (if applicable)

    def triggers(self, year: str, data_only: bool = False, mc_only: bool = False):
        """Get triggers for a given year for this channel."""
        return HLTs.hlts_by_type(year, self.hlt_types, data_only, mc_only)

    def lepton_triggers(self, year: str, data_only: bool = False, mc_only: bool = False):
        """Get lepton triggers for a given year for this channel."""
        if self.lepton_dataset is None:
            return None

        return HLTs.hlts_by_dataset(year, self.lepton_dataset, data_only, mc_only)


def parse_common_run_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="processor",
        type=str,
        choices=["skimmer"],
    )

    parser.add_argument(
        "--region",
        help="region",
        default="signal",
        choices=["signal"],
        type=str,
    )

    parser.add_argument(
        "--nano-version",
        type=str,
        default="v12_private",
        choices=["v12_private"],
        help="NanoAOD version",
    )
