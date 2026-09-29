"""Exportação da sessão para Parquet e JSON.

Cada frequência de amostragem vira um arquivo Parquet com uma coluna de tempo e
uma coluna por canal: canais da mesma frequência compartilham a base de tempo,
então não há reamostragem nem perda. Os metadados vão num JSON ao lado, com o
SHA-256 do .ld original para rastreabilidade.

Há duas colunas de tempo: `log_time` é o tempo do log (índice da amostra / frequência),
que não é tempo real porque o painel interrompe a gravação sem marcar; `gps_time` é
o horário absoluto do GPS, em UTC, nulo onde não dá para saber.

Nomes de coluna e chaves dos metadados são identificadores, em inglês. Os Parquet são gravados em
blocos de linhas, para a memória não crescer com o tamanho do log.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .lap_statistics import lap_statistics_table
from .session import Session

# 3: o tipo de volta passou a ser o identificador em inglês (out, timed, interrupted, incomplete).
# 4: colunas log_time e gps_time, metadata.json com chaves em inglês, fonte da
#    interrupção como identificador (gps, lap_timer) e lap_statistics.parquet.
# 5: motor, equipe e identificação do veículo no cabeçalho; os textos do cabeçalho como vieram,
#    sem aparar.
SCHEMA_VERSION = 5

LOG_TIME_COLUMN = "log_time"
GPS_TIME_COLUMN = "gps_time"
METADATA_FILE_NAME = "metadata.json"
LAP_STATISTICS_FILE_NAME = "lap_statistics.parquet"
ROWS_PER_BATCH = 32768
PARQUET_COMPRESSION = "zstd"


def frequency_file_name(frequency: int) -> str:
    return f"{frequency}Hz.parquet"


def _iso(moment: np.datetime64) -> str | None:
    return None if np.isnat(moment) else f"{moment}Z"


def session_metadata(session: Session) -> dict:
    header = session.header
    fastest = session.fastest_lap()
    laps = session.laps
    lap_starts = session.absolute_time(np.array([lap.start for lap in laps]))
    lap_ends = session.absolute_time(np.array([lap.end for lap in laps]))
    start_time = session.start_time
    return {
        "schema_version": SCHEMA_VERSION,
        "original_name": session.path.name,
        "sha256": session.sha256(),
        "size_in_bytes": session.structure.file_size,
        "device": {"type": header.device_type, "serial_number": header.device_serial,
                   "version": header.device_version},
        # Primeiro horário válido do GPS, em UTC. O relógio do painel e os textos do
        # cabeçalho não são confiáveis e ficam só como registro: nada é calculado com eles.
        "start_gps_time": start_time.isoformat().replace("+00:00", "Z") if start_time else None,
        "header": {"dash_clock": header.start.isoformat() if header.start else None,
                   "driver": header.driver, "vehicle": header.vehicle, "engine": header.engine,
                   "venue": header.venue, "team": header.team,
                   "vehicle_identification": header.vehicle_identification,
                   "event": header.event, "session": header.session,
                   "short_comment": header.short_comment},
        "duration_in_seconds": session.duration,
        "structure": {
            "channel_count": session.structure.channel_count,
            "data_ends_at_file_end": session.structure.data_ends_at_file_end,
            "gaps": len(session.structure.gaps),
            "warnings": session.structure.warnings,
        },
        "data_dropouts": [asdict(window) for window in session.dropouts],
        "recording_interruptions": [dict(asdict(interruption), located=interruption.located)
                                    for interruption in session.interruptions],
        "laps": [dict(asdict(lap), discrepancy=lap.discrepancy,
                      start_gps_time=_iso(start), end_gps_time=_iso(end))
                 for lap, start, end in zip(laps, lap_starts, lap_ends)],
        "fastest_lap": asdict(fastest) if fastest else None,
        "channels": [
            {"name": info.name, "short_name": info.short_name, "unit": info.unit,
             "frequency": info.frequency, "sample_count": info.sample_count,
             "data_type": info.numpy_type}
            for info in session.channels.values()
        ],
    }


def channels_by_frequency(session: Session) -> dict[int, list[str]]:
    by_frequency: dict[int, list[str]] = defaultdict(list)
    for name, info in session.channels.items():
        by_frequency[info.frequency].append(name)
    return dict(sorted(by_frequency.items()))


def frequency_schema(names: list[str]) -> pa.Schema:
    return pa.schema([(LOG_TIME_COLUMN, pa.float64()), (GPS_TIME_COLUMN, pa.timestamp("ms", tz="UTC"))]
                     + [(name, pa.float64()) for name in names])


def frequency_batches(session: Session, frequency: int, names: list[str],
                      rows_per_batch: int = ROWS_PER_BATCH) -> Iterator[pa.RecordBatch]:
    """Blocos de linhas de uma frequência: tempo do log, horário do GPS e os canais."""
    # Canais da mesma frequência podem, em tese, ter contagens diferentes:
    # completa com ausente em vez de truncar (truncar descartaria dado em silêncio).
    length = max(session.channels[name].sample_count for name in names)
    schema = frequency_schema(names)
    for first in range(0, length, rows_per_batch):
        last = min(first + rows_per_batch, length)
        time = np.arange(first, last) / frequency
        columns = [pa.array(time, type=pa.float64()),
                   pa.array(session.absolute_time(time), type=pa.timestamp("ms", tz="UTC"))]
        for name in names:
            values = np.full(last - first, np.nan)
            data = session.channel(name, first_sample=first, last_sample=last).values
            values[: len(data)] = data
            columns.append(pa.array(values, type=pa.float64(), from_pandas=True))
        yield pa.record_batch(columns, schema=schema)


def write_frequency_file(session: Session, frequency: int, names: list[str], path: Path) -> None:
    with pq.ParquetWriter(path, frequency_schema(names), compression=PARQUET_COMPRESSION) as writer:
        for batch in frequency_batches(session, frequency, names):
            writer.write_batch(batch)


def export_session(session: Session, output_directory: str | Path) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for frequency, names in channels_by_frequency(session).items():
        write_frequency_file(session, frequency, names, output / frequency_file_name(frequency))
    pq.write_table(lap_statistics_table(session), output / LAP_STATISTICS_FILE_NAME,
                   compression=PARQUET_COMPRESSION)
    with open(output / METADATA_FILE_NAME, "w", encoding="utf-8") as handle:
        json.dump(session_metadata(session), handle, ensure_ascii=False, indent=2, default=str)
    return output
