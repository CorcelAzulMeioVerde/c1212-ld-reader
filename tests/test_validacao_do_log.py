"""check_dash_log: só passa o log íntegro do painel C1212."""

import os

import pytest

from ld_sintetico import write_ld
from motec_dash import InvalidStructureError, NotADashLogError, NotAnLdFileError, check_dash_log


def test_log_do_painel_passa(ld_path):
    header = check_dash_log(ld_path)
    assert header.device_type == "C1212"


def test_log_de_outro_dispositivo_e_recusado(tmp_path):
    path = write_ld(tmp_path / "ecu.ld", device_type="M1")
    with pytest.raises(NotADashLogError, match="log de M1"):
        check_dash_log(path)


@pytest.mark.parametrize("fraction", [0.05, 0.5, 0.999])
def test_log_truncado_e_recusado(tmp_path, ld_path, fraction):
    copy = tmp_path / "truncado.ld"
    content = ld_path.read_bytes()
    copy.write_bytes(content[:int(len(content) * fraction)])
    with pytest.raises(InvalidStructureError, match="estrutura inválida"):
        check_dash_log(copy)


@pytest.mark.parametrize("content", [b"", b"\x40", b"<?xml version=\"1.0\"?><LDXFile/>" + b" " * 4000,
                                     os.urandom(100_000)])
def test_arquivo_que_nao_e_ld_e_recusado(tmp_path, content):
    path = tmp_path / "arquivo.ld"
    path.write_bytes(content)
    with pytest.raises(NotAnLdFileError, match="não é um log .ld"):
        check_dash_log(path)
