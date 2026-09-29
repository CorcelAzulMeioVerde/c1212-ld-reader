"""Gerador do rascunho do dicionário de canais (dicionario/canais.yaml).

O arquivo tem dois donos. A máquina é dona de `observado` (em cada canal) e do
bloco `sessoes`: o que os logs reais mostram, recalculado a cada execução. O
humano é dono de todo o resto: nome canônico, descrição, significado dos
estados, limites, prioridade e qualquer chave ou comentário que
acrescentar. Regenerar nunca altera o que é do humano; por isso a atualização é
feita no lugar, chave a chave, em vez de regravar o canal inteiro (regravar
perderia os comentários, que o YAML prende ao último valor do bloco anterior).

As estatísticas usam `Session.channel`, então já respeitam as regras de dado
ausente do leitor (-1 dentro de perda de comunicação, GPS sem posição).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

from .binary_format import LdFormatError
from .session import Session
from .validation import DASH_DEVICE_TYPE

SCHEMA_VERSION = 1
MAXIMUM_DISCRETE_VALUES = 20
ROUNDING_DECIMAL_PLACES = 6
HUMAN_FIELDS = ("nome_canonico", "descricao", "significado_dos_estados", "limites", "prioridade")
OBSERVED_KEY = "observado"
# Campos do cabeçalho do canal conferidos entre sessões: (chave no YAML, atributo de ChannelInfo).
COMPARED_CHANNEL_FIELDS = (("nome_curto", "short_name"), ("unidade", "unit"),
                           ("frequencia", "frequency"), ("tipo_de_dado", "numpy_type"))


class DictionaryFormatError(Exception):
    """Dicionário existente ilegível ou fora do formato: nada é gravado."""


@dataclass(frozen=True)
class RejectedLog:
    path: Path
    reason: str


@dataclass
class GenerationReport:
    output_path: Path
    accepted: list[Path] = field(default_factory=list)
    rejected: list[RejectedLog] = field(default_factory=list)
    channel_count: int = 0
    channels_missing_from_logs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- estatísticas

def _native(value: float) -> int | float:
    """Número nativo do Python, para o YAML não carregar tipos do numpy."""
    rounded = round(float(value), ROUNDING_DECIMAL_PLACES)
    return int(rounded) if rounded.is_integer() else rounded


def compute_statistics(values: np.ndarray) -> dict:
    """Estatísticas sobre as amostras válidas; NaN é amostra ausente."""
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    statistics = {"minimo": None, "maximo": None, "percentil_1": None, "percentil_99": None,
                  "fracao_de_amostras_ausentes": None, "numero_de_amostras": int(values.size)}
    if values.size:
        statistics["fracao_de_amostras_ausentes"] = _native(1 - valid.size / values.size)
    if valid.size:
        percentile_1, percentile_99 = np.percentile(valid, [1, 99])
        statistics.update(minimo=_native(valid.min()), maximo=_native(valid.max()),
                          percentil_1=_native(percentile_1), percentil_99=_native(percentile_99))
    return statistics


def compute_discrete_values(values: np.ndarray, frequency: float) -> dict[float, float] | None:
    """Tempo em segundos em cada valor distinto; None se passar do limite de canal discreto.

    Amostra ausente não conta tempo em nenhum valor.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.round(values[np.isfinite(values)], ROUNDING_DECIMAL_PLACES)
    distinct, counts = np.unique(valid, return_counts=True)
    if len(distinct) > MAXIMUM_DISCRETE_VALUES:
        return None
    return {float(value): float(count) / frequency for value, count in zip(distinct, counts)}


def combine_discrete_values(per_session: list[dict[float, float] | None]) -> dict[float, float] | None:
    """Soma os tempos entre sessões; o limite de valores distintos vale para o conjunto."""
    total: dict[float, float] = {}
    for session_values in per_session:
        if session_values is None:
            return None
        for value, seconds in session_values.items():
            total[value] = total.get(value, 0.0) + seconds
    if len(total) > MAXIMUM_DISCRETE_VALUES:
        return None
    return dict(sorted(total.items()))


def is_constant_in_all_sessions(per_session_statistics: list[dict]) -> bool | None:
    """None quando nenhuma sessão tem amostra válida do canal."""
    with_data = [statistics for statistics in per_session_statistics
                 if statistics["minimo"] is not None]
    if not with_data:
        return None
    return all(statistics["minimo"] == statistics["maximo"] for statistics in with_data)


# ------------------------------------------------------------ leitura dos logs

def find_dash_logs(directory: Path) -> tuple[list[Path], list[RejectedLog]]:
    """Logs do painel na pasta e subpastas, sem cópias repetidas (mesmo SHA-256)."""
    accepted: list[Path] = []
    rejected: list[RejectedLog] = []
    seen: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.ld")):
        try:
            with Session(path) as session:
                if session.header.device_type != DASH_DEVICE_TYPE:
                    rejected.append(RejectedLog(path, f"dispositivo {session.header.device_type!r}"
                                                      f" não é o painel {DASH_DEVICE_TYPE}"))
                    continue
                digest = session.sha256()
        except (LdFormatError, OSError) as error:
            rejected.append(RejectedLog(path, str(error)))
            continue
        if digest in seen:
            rejected.append(RejectedLog(path, f"cópia de {seen[digest].relative_to(directory)}"))
            continue
        seen[digest] = path
        accepted.append(path)
    return accepted, rejected


def describe_sessions(sessions: dict[str, Session]) -> dict[str, dict]:
    return {
        label: {
            "sha256": session.sha256(),
            "inicio": (session.start_time.isoformat().replace("+00:00", "Z")
                       if session.start_time else None),
            "duracao_em_segundos": _native(session.duration),
            "numero_de_canais": len(session.channels),
        }
        for label, session in sessions.items()
    }


def observe_channels(sessions: dict[str, Session]) -> dict[str, dict]:
    """Bloco `observado` de cada canal, combinando todas as sessões (rótulo -> sessão).

    O laço externo é por canal: só um canal, de todas as sessões, fica em memória por vez.
    """
    names: dict[str, None] = {}
    for session in sessions.values():
        names.update(dict.fromkeys(session.channels))

    observations: dict[str, dict] = {}
    for name in names:
        present = {label: session for label, session in sessions.items()
                   if name in session.channels}
        reference = next(iter(present.values())).channels[name]
        divergences = []
        for key, attribute in COMPARED_CHANNEL_FIELDS:
            found = {label: getattr(session.channels[name], attribute)
                     for label, session in present.items()}
            if len(set(found.values())) > 1:
                divergences.append({"campo": key, "valor_por_sessao": found})

        per_session_statistics: dict[str, dict] = {}
        per_session_discrete: list[dict[float, float] | None] = []
        all_values: list[np.ndarray] = []
        for label, session in present.items():
            data = session.channel(name)
            per_session_statistics[label] = compute_statistics(data.values)
            per_session_discrete.append(compute_discrete_values(data.values, data.info.frequency))
            all_values.append(data.values)
        discrete = combine_discrete_values(per_session_discrete)

        observed = {
            "presente_nos_logs": True,
            "nome_original": name,
            "nome_curto": reference.short_name,
            "unidade": reference.unit,
            "frequencia": reference.frequency,
            "tipo_de_dado": reference.numpy_type,
            "divergencias_entre_sessoes": divergences,
            "sessoes_sem_o_canal": [label for label in sessions if label not in present],
            "estatisticas": {
                "total": compute_statistics(np.concatenate(all_values)),
                "por_sessao": per_session_statistics,
            },
        }
        if discrete is not None:
            observed["valores_discretos"] = [
                {"valor": _native(value), "tempo_total_em_segundos": _native(seconds)}
                for value, seconds in discrete.items()]
        # Fica por último de propósito: ver _update_in_place.
        observed["constante_em_todas_as_sessoes"] = is_constant_in_all_sessions(
            list(per_session_statistics.values()))
        observations[name] = observed
    return observations


# ------------------------------------------------------------ mescla e gravação

def _empty_human_field(name: str):
    if name == "significado_dos_estados":
        return CommentedMap()
    if name == "limites":
        # Um único conjunto de limites.
        return CommentedMap(minimo=None, maximo=None)
    return None


def _to_commented(value):
    if isinstance(value, dict):
        return CommentedMap((key, _to_commented(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_to_commented(item) for item in value]
    return value


def _update_in_place(target: CommentedMap, new: dict) -> None:
    """Deixa `target` igual a `new` sem trocar o objeto nem a ordem das chaves.

    O YAML prende um comentário escrito antes de uma chave ao último valor do
    bloco anterior. Atribuir chave a chave mantém esses comentários; por isso a
    última chave de `observado` é sempre a mesma e é um valor simples.
    """
    for key in [key for key in target if key not in new]:
        del target[key]
    for position, (key, value) in enumerate(new.items()):
        if key not in target:
            target.insert(position, key, _to_commented(value))
        elif isinstance(value, dict) and isinstance(target[key], CommentedMap):
            _update_in_place(target[key], value)
        else:
            target[key] = _to_commented(value)


def merge_observations(document: CommentedMap | None, observations: dict[str, dict],
                       sessions: dict[str, dict]) -> CommentedMap:
    """Atualiza só `observado` e `sessoes`; tudo o mais no documento é do humano."""
    if document is None:
        document = CommentedMap()
    if not isinstance(document, CommentedMap):
        raise DictionaryFormatError("a raiz do dicionário deveria ser um mapeamento")
    document["versao_do_esquema"] = SCHEMA_VERSION
    channels = document.setdefault("canais", CommentedMap())
    if not isinstance(channels, CommentedMap):
        raise DictionaryFormatError("'canais' deveria ser um mapeamento de nome de canal")
    for name, entry in channels.items():
        if not isinstance(entry, CommentedMap):
            raise DictionaryFormatError(f"canal '{name}' deveria ser um mapeamento")

    for name, entry in channels.items():
        if name not in observations:
            # Mantém o preenchimento humano de um canal que saiu dos logs.
            observations = {**observations, name: {"presente_nos_logs": False,
                                                   "constante_em_todas_as_sessoes": None}}
    for name, observed in observations.items():
        entry = channels.setdefault(name, CommentedMap())
        for field_name in HUMAN_FIELDS:
            if field_name not in entry:
                position = (list(entry).index(OBSERVED_KEY) if OBSERVED_KEY in entry
                            else len(entry))
                entry.insert(position, field_name, _empty_human_field(field_name))
        if isinstance(entry.get(OBSERVED_KEY), CommentedMap):
            _update_in_place(entry[OBSERVED_KEY], observed)
        else:
            entry[OBSERVED_KEY] = _to_commented(observed)

    # `sessoes` fica no fim do arquivo: assim nenhum comentário dos canais se prende a ele.
    if isinstance(document.get("sessoes"), CommentedMap):
        _update_in_place(document["sessoes"], sessions)
    else:
        document["sessoes"] = _to_commented(sessions)
    return document


def dictionary_yaml() -> YAML:
    """O YAML do dicionário: o mesmo na gravação e na releitura."""
    yaml = YAML()
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def load_dictionary(path: Path) -> CommentedMap | None:
    if not path.exists():
        return None
    try:
        document = dictionary_yaml().load(path.read_text(encoding="utf-8"))
    except YAMLError as error:
        raise DictionaryFormatError(f"{path} não é um YAML válido: {error}") from error
    return document


def save_dictionary(document: CommentedMap, path: Path) -> None:
    """Grava em arquivo temporário e troca no fim: falha no meio não corrompe o dicionário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            dictionary_yaml().dump(document, stream)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_dictionary(samples_directory: str | Path, output_path: str | Path) -> GenerationReport:
    samples_directory, output_path = Path(samples_directory), Path(output_path)
    # Lê o existente antes do trabalho pesado: se estiver inválido, para logo.
    document = load_dictionary(output_path)
    accepted, rejected = find_dash_logs(samples_directory)
    if not accepted:
        raise FileNotFoundError(f"nenhum log do painel {DASH_DEVICE_TYPE} em {samples_directory}")

    sessions = {path.relative_to(samples_directory).as_posix(): Session(path) for path in accepted}
    try:
        observations = observe_channels(sessions)
        descriptions = describe_sessions(sessions)
    finally:
        for session in sessions.values():
            session.close()

    document = merge_observations(document, observations, descriptions)
    save_dictionary(document, output_path)
    missing = [name for name, entry in document["canais"].items()
               if not entry[OBSERVED_KEY]["presente_nos_logs"]]
    return GenerationReport(output_path=output_path, accepted=accepted, rejected=rejected,
                            channel_count=len(document["canais"]),
                            channels_missing_from_logs=missing)
