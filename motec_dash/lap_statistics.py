"""Estatísticas por volta: uma linha por volta e canal.

Mínimo, máximo, média e quantidade de amostras válidas. Dado ausente nunca entra:
os valores vêm de Session.channel, que já marca como ausente o que está dentro de
perda de dados e o GPS sem posição. Uma amostra é da volta se início <= tempo do
log < fim. Volta e canal sem nenhuma amostra válida têm a linha com quantidade zero
e estatísticas nulas: assim "sem dado nesta volta" fica explícito.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from .session import Session

LAP_STATISTICS_SCHEMA = pa.schema([
    ("sequence_number", pa.int32()), ("channel", pa.string()), ("minimum", pa.float64()),
    ("maximum", pa.float64()), ("mean", pa.float64()), ("valid_sample_count", pa.int64())])


def lap_sample_ranges(session: "Session", frequency: int, sample_count: int) -> list[tuple[int, int]]:
    """(primeira amostra, última amostra exclusiva) de cada volta, nesta frequência."""
    time = np.arange(sample_count) / frequency
    return [(int(np.searchsorted(time, lap.start, side="left")),
             int(np.searchsorted(time, lap.end, side="left"))) for lap in session.laps]


def lap_statistics_table(session: "Session") -> pa.Table:
    laps = session.laps
    columns: dict[str, list] = {field.name: [] for field in LAP_STATISTICS_SCHEMA}
    ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for name, info in session.channels.items():
        key = (info.frequency, info.sample_count)
        if key not in ranges:
            ranges[key] = lap_sample_ranges(session, *key)
        values = session.channel(name).values
        for lap, (first, last) in zip(laps, ranges[key]):
            valid = values[first:last]
            valid = valid[np.isfinite(valid)]
            columns["sequence_number"].append(lap.number)
            columns["channel"].append(name)
            columns["valid_sample_count"].append(len(valid))
            columns["minimum"].append(float(valid.min()) if len(valid) else None)
            columns["maximum"].append(float(valid.max()) if len(valid) else None)
            columns["mean"].append(float(valid.mean()) if len(valid) else None)
    return pa.table(columns, schema=LAP_STATISTICS_SCHEMA)
