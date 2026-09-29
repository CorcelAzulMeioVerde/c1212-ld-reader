"""Leitor contra o .ld sintético: estrutura, dado ausente, voltas, interrupções e exportação."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

import ld_sintetico as sintetico
from motec_dash import LAP_KIND_TEXT, LAP_KINDS
from motec_dash.__main__ import print_summary
from motec_dash.export import SCHEMA_VERSION, export_session
from motec_dash.lap_statistics import lap_statistics_table


def test_estrutura_integra(session):
    structure = session.structure
    assert structure.channel_count == len(sintetico.build_channels())
    assert structure.gaps == []
    assert structure.data_ends_at_file_end
    assert structure.warnings == []


def test_cabecalho(session):
    header = session.header
    assert header.device_type == "C1212"
    assert header.device_serial == sintetico.DEVICE_SERIAL
    assert {name: getattr(header, name) for name in sintetico.HEADER} == sintetico.HEADER


def test_valores_escalados(session):
    speed = session.channel("Ground Speed")
    expected = 120 + 60 * np.sin(speed.time / 9)
    valid = np.isfinite(speed.values)
    assert np.allclose(speed.values[valid], expected[valid], atol=0.05)
    assert speed.info.frequency == 20 and speed.info.unit == "km/h"


def test_perda_de_dados_vira_ausente_e_reinicio_da_ecu(session):
    [window] = session.dropouts
    assert window.start <= sintetico.DROPOUT[0] and window.end >= sintetico.DROPOUT[1]
    assert window.ecu_restarted is True
    assert window.ground_speed_before is not None
    engine = session.channel("Engine Speed")
    inside = (engine.time >= sintetico.DROPOUT[0]) & (engine.time < sintetico.DROPOUT[1])
    assert np.all(np.isnan(engine.values[inside]))
    assert np.nanmin(engine.values) >= 0, "sem dado vazou como rotação negativa"


def test_menos_um_legitimo_preservado(session):
    raw = session.raw("G Force Lat")
    values = session.channel("G Force Lat").values
    legitimate = (raw == -1) & np.isfinite(values)
    assert legitimate.sum() > 0
    assert np.allclose(values[legitimate], -0.01)


def test_gps_sem_posicao_vira_ausente(session):
    quality = session.channel("GPS Quality").values
    for name in ("GPS Latitude", "GPS Longitude", "GPS Speed"):
        assert np.all(np.isnan(session.channel(name).values[quality == 0])), name
    assert not np.any(session.channel("GPS Latitude").values == 0)


def test_voltas(session):
    assert [lap.kind for lap in session.laps] == ["out", "timed", "timed", "interrupted", "incomplete"]
    assert [lap.number for lap in session.laps] == [0, 1, 2, 3, 4]
    for lap in session.timed_laps():
        assert lap.precise
        assert lap.official_time == pytest.approx(100.0)
        assert abs(lap.discrepancy) <= 0.02
    interrupted = session.laps[3]
    assert interrupted.official_time == pytest.approx(120.0)
    assert interrupted.measured_duration + sintetico.INTERRUPTION_SECONDS == pytest.approx(
        interrupted.official_time, abs=0.02)
    assert session.fastest_lap().number == 1


def test_interrupcao_de_gravacao_pelo_gps(session):
    [interruption] = session.interruptions
    assert interruption.source == "gps" and interruption.located
    assert interruption.missing_seconds == pytest.approx(sintetico.INTERRUPTION_SECONDS, abs=0.15)
    assert interruption.start < sintetico.INTERRUPTION_AT <= interruption.end


def test_horario_absoluto(session):
    """O início vem do GPS (o relógio do painel no cabeçalho não é usado), com a correção
    das 1024 semanas; depois da interrupção, o horário salta o tempo sem gravação."""
    expected_start = sintetico.START_UTC + np.timedelta64(int(sintetico.NO_FIX_UNTIL), "s")
    assert np.datetime64(session.start_time.replace(tzinfo=None), "s") == expected_start
    assert session.start_time.utcoffset().total_seconds() == 0
    before, after = session.absolute_time(np.array([200.0, 350.0]))
    start_ms = sintetico.START_UTC.astype("datetime64[ms]")
    assert abs((before - start_ms) / np.timedelta64(1, "ms") - 200_000) <= 100
    assert abs((after - start_ms) / np.timedelta64(1, "ms") - 410_000) <= 100
    known = session.absolute_time(np.arange(int(session.duration * 20)) / 20)
    known = known[~np.isnat(known)].astype(np.int64)
    assert np.all(np.diff(known) >= 0)


def test_exportacao(session, tmp_path):
    output = export_session(session, tmp_path / "saida")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert len(metadata["sha256"]) == 64
    assert len(metadata["channels"]) == session.structure.channel_count
    assert [lap["kind"] for lap in metadata["laps"]] == [lap.kind for lap in session.laps]
    assert len(metadata["recording_interruptions"]) == 1
    table = pq.read_table(output / "100Hz.parquet")
    throttle = session.channel("Throttle Position")
    assert table.column_names[:2] == ["log_time", "gps_time"]
    assert str(table.schema.field("gps_time").type) == "timestamp[ms, tz=UTC]"
    assert np.array_equal(table["Throttle Position"].to_numpy(), throttle.values, equal_nan=True)
    assert np.array_equal(pq.read_table(output / "20Hz.parquet")["GPS Speed"].to_numpy(),
                          session.channel("GPS Speed").values, equal_nan=True)


def test_estatisticas_por_volta_contra_numpy(session):
    table = lap_statistics_table(session)
    assert table.num_rows == len(session.laps) * len(session.channels)
    found = {(row["sequence_number"], row["channel"]): row for row in table.to_pylist()}
    for name in session.channels:
        data = session.channel(name)
        for lap in session.laps:
            inside = (data.time >= lap.start) & (data.time < lap.end) & np.isfinite(data.values)
            row, values = found[(lap.number, name)], data.values[inside]
            assert row["valid_sample_count"] == int(inside.sum())
            if not len(values):
                assert row["minimum"] is row["maximum"] is row["mean"] is None
                continue
            assert (row["minimum"], row["maximum"]) == (values.min(), values.max())
            assert row["mean"] == pytest.approx(values.mean(), rel=1e-12, abs=1e-12)


def test_resumo(session, capsys):
    print_summary(session)
    shown = capsys.readouterr().out
    assert f"Início: {session.start_time:%d/%m/%Y %H:%M:%S} UTC (GPS)" in shown
    assert set(LAP_KIND_TEXT) == set(LAP_KINDS)
    for lap in session.laps:
        assert LAP_KIND_TEXT[lap.kind] in shown
