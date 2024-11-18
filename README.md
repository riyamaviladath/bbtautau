# HHbbtautau


[![Actions Status][actions-badge]][actions-link]
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/LPC-HH/bbtautau/main.svg)](https://results.pre-commit.ci/latest/github/LPC-HH/bbtautau/main)
<!-- [![Documentation Status][rtd-badge]][rtd-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LPC-HH/bbtautau/workflows/CI/badge.svg
[actions-link]:             https://github.com/LPC-HH/bbtautau/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/bbtautau
[conda-link]:               https://github.com/conda-forge/bbtautau-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/LPC-HH/bbtautau/discussions
[pypi-link]:                https://pypi.org/project/bbtautau/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/bbtautau
[pypi-version]:             https://img.shields.io/pypi/v/bbtautau
[rtd-badge]:                https://readthedocs.org/projects/bbtautau/badge/?version=latest
[rtd-link]:                 https://bbtautau.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to two beauty quarks (b) and two tau leptons.


- [HHbbtautau](#hhbbtautau)
  - [Setting up package](#setting-up-package)
    - [Creating a virtual environment](#creating-a-virtual-environment)
    - [Installing package](#installing-package)
    - [Troubleshooting](#troubleshooting)
  - [Running coffea processors](#running-coffea-processors)
    - [Setup](#setup)


## Setting up package

### Creating a virtual environment

First, create a virtual environment (`micromamba` is recommended):

```bash
# Clone the repository
git clone --recursive https://github.com/LPC-HH/bbtautau.git
cd bbtautau
# Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
# Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
micromamba env create -f environment.yaml
micromamba activate hh
```

### Installing package

**Remember to install this in your mamba environment**.

```bash
# Clone the repsitory as above if you haven't already
# Perform an editable installation
pip install -e .
# for committing to the repository
pip install pre-commit
pre-commit install
# Install as well the common HH utilities
cd boostedhh
pip install -e .
cd ..
```

### Troubleshooting

- If your default `python` in your environment is not Python 3, make sure to use
  `pip3` and `python3` commands instead.

- You may also need to upgrade `pip` to perform the editable installation:

```bash
python3 -m pip install -e .
```

## Running coffea processors

### Setup

For submitting to condor, all you need is python >= 3.7.

For running locally, follow the same virtual environment setup instructions
above and install `coffea`

```bash
micromamba activate hh
pip install coffea
```

Clone the repository:

```
git clone https://github.com/LPC-HH/bbtautau/
pip install -e .
```

### Running locally

For testing, e.g.:

```bash
python src/run.py --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8 --starti 0 --endi 1 --year 2022 --processor skimmer
```

### Condor jobs

A single sample / subsample:

```bash
python src/condor/submit.py --analysis bbtautau --git-branch signal_study --site ucsd --save-sites ucsd lpc --processor skimmer --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8 --files-per-job 5 --tag 24Nov7Signal --submit
```

## Transferring files to FNAL with Rucio

Set up Rucio following the [Twiki](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookFileTransfer#Option_1_Using_the_Command_L_AN1). Then:

```bash
rucio add-rule cms:/Tau/Run2022F-22Sep2023-v1/MINIAOD 1 T1_US_FNAL_Disk --activity "User AutoApprove" --lifetime 15552000 --ask-approval
```
