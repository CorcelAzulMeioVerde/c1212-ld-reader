"""Fixtures: um .ld sintético (ver ld_sintetico.py), gerado uma vez por execução."""

from __future__ import annotations

import pytest

from ld_sintetico import write_ld
from motec_dash import Session


@pytest.fixture(scope="session")
def ld_path(tmp_path_factory):
    return write_ld(tmp_path_factory.mktemp("amostra") / "sintetico.ld")


@pytest.fixture
def session(ld_path):
    with Session(ld_path) as opened:
        yield opened
