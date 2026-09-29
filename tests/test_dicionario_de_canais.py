"""Testes do gerador do dicionário de canais, com dados sintéticos."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from motec_dash import channel_dictionary as cd

from ld_sintetico import write_ld


# ---------------------------------------------------------------- estatísticas

def test_estatisticas_ignoram_amostras_ausentes():
    values = np.concatenate([np.arange(1.0, 101.0), [np.nan] * 25])
    statistics = cd.compute_statistics(values)
    assert statistics["minimo"] == 1
    assert statistics["maximo"] == 100
    assert statistics["percentil_1"] == pytest.approx(1.99)
    assert statistics["percentil_99"] == pytest.approx(99.01)
    assert statistics["fracao_de_amostras_ausentes"] == pytest.approx(0.2)
    assert statistics["numero_de_amostras"] == 125


def test_estatisticas_de_canal_todo_ausente():
    statistics = cd.compute_statistics(np.full(10, np.nan))
    assert statistics["minimo"] is None and statistics["percentil_99"] is None
    assert statistics["fracao_de_amostras_ausentes"] == 1


def test_estatisticas_usam_numeros_nativos():
    """Tipo do numpy no dicionário viraria etiqueta ilegível no YAML."""
    statistics = cd.compute_statistics(np.array([0.5, 2.0], dtype=np.float32))
    assert all(type(value) in (int, float) for value in statistics.values())


def test_tempo_em_cada_valor_discreto():
    values = np.array([0, 0, 0, 0, 3, 3, 4, np.nan, np.nan, np.nan])
    assert cd.compute_discrete_values(values, frequency=2) == {0.0: 2.0, 3.0: 1.0, 4.0: 0.5}


def test_limite_de_valores_distintos():
    assert cd.compute_discrete_values(np.arange(20.0), frequency=1) is not None
    assert cd.compute_discrete_values(np.arange(21.0), frequency=1) is None


def test_tempo_discreto_soma_sessoes_de_frequencias_diferentes():
    first = cd.compute_discrete_values(np.array([1.0, 1.0, 2.0, 2.0]), frequency=1)
    second = cd.compute_discrete_values(np.array([1.0] * 10), frequency=10)
    assert cd.combine_discrete_values([first, second]) == {1.0: 3.0, 2.0: 2.0}


def test_limite_de_valores_distintos_vale_para_o_conjunto_das_sessoes():
    first = cd.compute_discrete_values(np.arange(0.0, 15.0), frequency=1)
    second = cd.compute_discrete_values(np.arange(10.0, 25.0), frequency=1)
    assert first is not None and second is not None
    assert cd.combine_discrete_values([first, second]) is None


def test_constante_so_se_for_em_todas_as_sessoes():
    constant = cd.compute_statistics(np.array([5.0, 5.0, np.nan]))
    other_constant = cd.compute_statistics(np.array([7.0, 7.0]))
    varying = cd.compute_statistics(np.array([5.0, 6.0]))
    absent = cd.compute_statistics(np.array([np.nan]))
    assert cd.is_constant_in_all_sessions([constant, other_constant, absent]) is True
    assert cd.is_constant_in_all_sessions([constant, varying]) is False
    assert cd.is_constant_in_all_sessions([absent]) is None


# ----------------------------------------------- preservação do preenchimento

def _observed(maximum: float, sessions: list[str]) -> dict:
    statistics = {"minimo": 0, "maximo": maximum, "percentil_1": 0, "percentil_99": maximum,
                  "fracao_de_amostras_ausentes": 0, "numero_de_amostras": 100}
    return {
        "presente_nos_logs": True, "nome_original": "X", "nome_curto": "X", "unidade": "%",
        "frequencia": 100, "tipo_de_dado": "<i2", "divergencias_entre_sessoes": [],
        "sessoes_sem_o_canal": [],
        "estatisticas": {"total": dict(statistics),
                         "por_sessao": {label: dict(statistics) for label in sessions}},
        "constante_em_todas_as_sessoes": False,
    }


def _sessions(labels: list[str]) -> dict:
    return {label: {"sha256": "0" * 64, "inicio": None, "duracao_em_segundos": 10,
                    "numero_de_canais": 2} for label in labels}


def _generate(path: Path, observations: dict, labels: list[str]):
    document = cd.merge_observations(cd.load_dictionary(path), observations, _sessions(labels))
    cd.save_dictionary(document, path)
    return cd.load_dictionary(path)


def _human_part(entry) -> dict:
    return {key: value for key, value in entry.items() if key != cd.OBSERVED_KEY}


def test_canal_novo_nasce_com_campos_humanos_vazios(tmp_path):
    path = tmp_path / "dicionario" / "canais.yaml"
    document = _generate(path, {"Throttle Position": _observed(63, ["a.ld"])}, ["a.ld"])
    entry = document["canais"]["Throttle Position"]
    assert list(entry) == [*cd.HUMAN_FIELDS, cd.OBSERVED_KEY]
    assert entry["nome_canonico"] is None and entry["descricao"] is None
    assert entry["prioridade"] is None
    assert entry["significado_dos_estados"] == {}
    assert entry["limites"] == {"minimo": None, "maximo": None}


def test_preenchimento_humano_nunca_e_sobrescrito(tmp_path):
    path = tmp_path / "canais.yaml"
    _generate(path, {"Throttle Position": _observed(63, ["a.ld"]),
                     "P2P State": _observed(4, ["a.ld"])}, ["a.ld"])

    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "  P2P State:\n    nome_canonico:\n    descricao:\n    significado_dos_estados: {}\n",
        "  # revisar com a MoTeC\n"
        "  P2P State:\n    nome_canonico: Estado do Push to Pass  # confirmado na pista\n"
        "    descricao: Máquina de estados do Push to Pass\n"
        "    significado_dos_estados:\n      3: contagem\n      4: ativo\n"
        "    responsavel: engenharia\n", 1)
    text = text.replace("    limites:\n      minimo:\n      maximo:\n",
                        "    limites:\n      minimo: 0\n      maximo: 63\n", 1)
    text = "# Dicionário de canais: editar só fora de 'observado'\n" + text
    path.write_text(text, encoding="utf-8")
    edited = cd.load_dictionary(path)
    assert edited["canais"]["P2P State"]["nome_canonico"] == "Estado do Push to Pass"
    assert edited["canais"]["Throttle Position"]["limites"]["maximo"] == 63
    expected = {name: _human_part(entry) for name, entry in edited["canais"].items()}

    regenerated = _generate(path, {"Throttle Position": _observed(99, ["a.ld", "b.ld"]),
                                   "P2P State": _observed(4, ["a.ld", "b.ld"])}, ["a.ld", "b.ld"])
    for name, entry in regenerated["canais"].items():
        assert _human_part(entry) == expected[name], name
    # As estatísticas, essas sim, foram atualizadas.
    observed = regenerated["canais"]["Throttle Position"][cd.OBSERVED_KEY]
    assert observed["estatisticas"]["total"]["maximo"] == 99
    assert list(observed["estatisticas"]["por_sessao"]) == ["a.ld", "b.ld"]
    assert list(regenerated["sessoes"]) == ["a.ld", "b.ld"]
    # Comentários também são preenchimento humano.
    new_text = path.read_text(encoding="utf-8")
    for comment in ("# Dicionário de canais: editar só fora de 'observado'",
                    "# revisar com a MoTeC", "# confirmado na pista"):
        assert comment in new_text, comment
    assert new_text.index("# revisar com a MoTeC") < new_text.index("  P2P State:")


def test_regenerar_sem_mudanca_nos_logs_nao_altera_o_arquivo(tmp_path):
    path = tmp_path / "canais.yaml"
    observations = {"Throttle Position": _observed(63, ["a.ld"])}
    _generate(path, observations, ["a.ld"])
    first = path.read_text(encoding="utf-8")
    _generate(path, observations, ["a.ld"])
    assert path.read_text(encoding="utf-8") == first


def test_canal_que_sumiu_dos_logs_mantem_o_preenchimento(tmp_path):
    path = tmp_path / "canais.yaml"
    _generate(path, {"Antigo": _observed(1, ["a.ld"]), "Atual": _observed(1, ["a.ld"])}, ["a.ld"])
    path.write_text(path.read_text(encoding="utf-8").replace(
        "  Antigo:\n    nome_canonico:\n", "  Antigo:\n    nome_canonico: Canal antigo\n"),
        encoding="utf-8")
    regenerated = _generate(path, {"Atual": _observed(2, ["b.ld"])}, ["b.ld"])
    old = regenerated["canais"]["Antigo"]
    assert old["nome_canonico"] == "Canal antigo"
    assert old[cd.OBSERVED_KEY]["presente_nos_logs"] is False
    assert "estatisticas" not in old[cd.OBSERVED_KEY]


def test_campo_humano_editado_parcialmente_fica_como_o_humano_deixou(tmp_path):
    """Apagar uma chave dos limites é decisão humana: o gerador não a recria."""
    path = tmp_path / "canais.yaml"
    _generate(path, {"Canal": _observed(1, ["a.ld"])}, ["a.ld"])
    text = path.read_text(encoding="utf-8").replace(
        "    limites:\n      minimo:\n      maximo:\n", "    limites:\n      maximo:\n")
    assert "      minimo:\n      maximo:" not in text
    path.write_text(text, encoding="utf-8")
    regenerated = _generate(path, {"Canal": _observed(2, ["a.ld"])}, ["a.ld"])
    assert list(regenerated["canais"]["Canal"]["limites"]) == ["maximo"]


@pytest.mark.parametrize("content", [
    "canais: [isto não é um mapeamento",
    "- uma lista\n- na raiz\n",
    "canais:\n  Throttle Position: texto solto\n",
])
def test_dicionario_invalido_nao_e_regravado(tmp_path, content):
    path = tmp_path / "canais.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(cd.DictionaryFormatError):
        _generate(path, {"Throttle Position": _observed(63, ["a.ld"])}, ["a.ld"])
    assert path.read_text(encoding="utf-8") == content
    assert list(tmp_path.iterdir()) == [path]


# ------------------------------------------------------------- log sintético

def test_dicionario_a_partir_de_logs(tmp_path, ld_path):
    """Cópia do mesmo arquivo e log de outro dispositivo ficam de fora; as estatísticas
    não levam o dado ausente (GPS sem posição, perda de dados) e mantêm o -1 legítimo."""
    samples = tmp_path / "amostras"
    (samples / "subpasta").mkdir(parents=True)
    shutil.copy(ld_path, samples / "a.ld")
    shutil.copy(ld_path, samples / "subpasta" / "copia.ld")
    write_ld(samples / "ecu.ld", device_type="M1")

    output = tmp_path / "canais.yaml"
    report = cd.write_dictionary(samples, output)
    assert [path.name for path in report.accepted] == ["a.ld"]
    reasons = sorted(rejected.reason for rejected in report.rejected)
    assert len(reasons) == 2 and reasons[0].startswith("cópia de") and "M1" in reasons[1]

    channels = cd.load_dictionary(output)["canais"]
    latitude = channels["GPS Latitude"]["observado"]["estatisticas"]["total"]
    assert latitude["fracao_de_amostras_ausentes"] > 0 and latitude["maximo"] < 0
    assert channels["G Force Lat"]["observado"]["estatisticas"]["total"]["minimo"] < -0.01
    quality = {item["valor"] for item in channels["GPS Quality"]["observado"]["valores_discretos"]}
    assert quality == {0, 2}

    before = output.read_text(encoding="utf-8")
    cd.write_dictionary(samples, output)
    assert output.read_text(encoding="utf-8") == before
