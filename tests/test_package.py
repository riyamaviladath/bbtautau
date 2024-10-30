from __future__ import annotations

import importlib.metadata

import bbtautau as m


def test_version():
    assert importlib.metadata.version("bbtautau") == m.__version__
