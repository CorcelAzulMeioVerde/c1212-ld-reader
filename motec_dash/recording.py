"""Interrupções de gravação e horário absoluto das amostras.

O painel interrompe a gravação e o arquivo não marca a interrupção: o tempo
calculado pelo índice da amostra ("tempo do log") não é tempo real. A referência é
o relógio do GPS ("GPS Date" + "GPS Time", em UTC), absoluto e comum a todos os
carros. A data e o horário do cabeçalho vêm do relógio do painel e não são usados.

Regras validadas nos logs reais:
  - "GPS Time" é hhmmss,s com resolução de 0,1 s, amostrado a 20 Hz. Só vale com
    posição resolvida: sem posição grava lixo (bruto que não é múltiplo de 100).
  - "GPS Date" é ddmmaa. Na maioria dos painéis sai 1024 semanas atrasado (estouro
    do contador de semanas do GPS) e pode valer 0 mesmo com posição resolvida.
  - Dentro de um trecho contínuo de gravação, horário do GPS - tempo do log fica
    estável dentro da resolução do canal.
  - O cronômetro ("Running Lap Time") também salta para a frente numa interrupção,
    mas só quando está contando: parado em zero ele não mostra nada. Por isso ele é
    só complemento, onde o GPS não tem posição.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .laps import find_timer_jumps

if TYPE_CHECKING:
    from .session import Session

GPS_CLOCK_CHANNELS = ("GPS Time", "GPS Date", "GPS Quality")
GPS_TIME_RAW_UNITS_PER_SECOND = 1000
GPS_TIME_RAW_RESOLUTION = 100                # 0,1 s: fora disso o valor é lixo
GPS_WEEK_ROLLOVER = np.timedelta64(1024 * 7, "D")
GPS_WEEK_ROLLOVER_BEFORE = np.datetime64("2020-01-01")
INTERRUPTION_MINIMUM_SECONDS = 1.0
# O salto do cronômetro confirma a interrupção vista pelo GPS se as durações batem.
TIMER_MATCH_TOLERANCE_SECONDS = 1.0
# Interrupção entre duas amostras vizinhas (do GPS ou do cronômetro) está localizada.
LOCATED_MAXIMUM_SECONDS = 0.1
# Trecho contínuo mais curto que isto usa deslocamento constante em vez de reta.
LINEAR_ANCHOR_MINIMUM_SECONDS = 60.0

# Identificadores da fonte da interrupção; o texto em português existe só para exibição.
SOURCE_GPS = "gps"
SOURCE_TIMER = "lap_timer"
INTERRUPTION_SOURCES = (SOURCE_GPS, SOURCE_TIMER)
INTERRUPTION_SOURCE_TEXT = {SOURCE_GPS: "GPS", SOURCE_TIMER: "cronômetro"}


@dataclass(frozen=True)
class RecordingInterruption:
    start: float            # tempo do log da última amostra antes da interrupção
    end: float              # tempo do log da primeira amostra depois
    missing_seconds: float  # duração real sem gravação
    source: str             # INTERRUPTION_SOURCES: "gps" ou "lap_timer"

    @property
    def located(self) -> bool:
        """Falso quando a interrupção caiu num trecho sem posição do GPS e nada mais
        a localiza: sabe-se quanto tempo faltou, mas não em que amostra."""
        return self.end - self.start <= LOCATED_MAXIMUM_SECONDS


@dataclass(frozen=True)
class GpsClock:
    time: np.ndarray        # tempo do log das amostras com horário válido
    seconds: np.ndarray     # horário UTC, em segundos desde 1970-01-01
    ticked: np.ndarray      # o valor de "GPS Time" acabou de mudar nesta amostra
    sample_period: float    # segundos entre amostras de "GPS Time" (0,05 nos logs reais)


def _dates_from_ddmmyy(raw_date: np.ndarray) -> np.ndarray:
    """ddmmaa -> datetime64[D], já com a correção das 1024 semanas. Inválida = NaT."""
    dates = np.full(len(raw_date), np.datetime64("NaT", "D"))
    for value in np.unique(raw_date):
        day, month, year = int(value) // 10000, int(value) // 100 % 100, int(value) % 100
        try:
            date = np.datetime64(f"20{year:02d}-{month:02d}-{day:02d}")
        except ValueError:
            continue
        if date < GPS_WEEK_ROLLOVER_BEFORE:
            date = date + GPS_WEEK_ROLLOVER
        dates[raw_date == value] = date
    return dates


def read_gps_clock(session: "Session") -> GpsClock | None:
    if not all(name in session.channels for name in GPS_CLOCK_CHANNELS):
        return None
    raw_time = session.raw("GPS Time").astype(np.int64)
    raw_date = session.raw("GPS Date").astype(np.int64)
    quality = session.raw("GPS Quality")
    if not len(raw_time) == len(raw_date) == len(quality):
        return None
    frequency = session.channels["GPS Time"].frequency

    whole = raw_time // GPS_TIME_RAW_UNITS_PER_SECOND
    hours, minutes, seconds = whole // 10000, whole // 100 % 100, whole % 100
    valid = ((quality > 0) & (raw_date > 0) & (raw_time >= 0)
             & (raw_time % GPS_TIME_RAW_RESOLUTION == 0)
             & (hours < 24) & (minutes < 60) & (seconds < 60))
    no_fix = session._gps_without_fix()
    if no_fix is not None and len(no_fix) == len(valid):
        valid &= ~no_fix

    index = np.flatnonzero(valid)
    dates = _dates_from_ddmmyy(raw_date[index])
    index, dates = index[~np.isnat(dates)], dates[~np.isnat(dates)]
    day_seconds = (hours[index] * 3600 + minutes[index] * 60 + seconds[index]
                   + (raw_time[index] % GPS_TIME_RAW_UNITS_PER_SECOND)
                   / GPS_TIME_RAW_UNITS_PER_SECOND)
    epoch_seconds = dates.astype("datetime64[s]").astype(np.int64) + day_seconds
    ticked = np.zeros(len(index), dtype=bool)
    if len(index) > 1:
        ticked[1:] = (np.diff(index) == 1) & (np.diff(raw_time[index]) != 0)
    return GpsClock(time=index / frequency, seconds=epoch_seconds, ticked=ticked,
                    sample_period=1 / frequency)


def find_interruptions(session: "Session") -> list[RecordingInterruption]:
    clock = session.gps_clock
    interruptions: list[RecordingInterruption] = []
    if clock is not None and len(clock.time) > 1:
        excess = np.diff(clock.seconds) - np.diff(clock.time)
        for index in np.flatnonzero(excess > INTERRUPTION_MINIMUM_SECONDS):
            interruptions.append(RecordingInterruption(
                start=float(clock.time[index]), end=float(clock.time[index + 1]),
                missing_seconds=float(excess[index]), source=SOURCE_GPS))

    if "Running Lap Time" not in session.channels:
        return interruptions
    running = session.channel("Running Lap Time")
    sample_period = 1 / running.info.frequency
    for jump_time, missing in find_timer_jumps(running):
        if clock is not None and len(clock.time):
            after = int(np.searchsorted(clock.time, jump_time - sample_period / 2))
            if 0 < after < len(clock.time):
                # O GPS tem horário dos dois lados do salto: ele decide se houve interrupção.
                _locate_with_timer(interruptions, jump_time, sample_period, missing)
                continue
        interruptions.append(RecordingInterruption(
            start=jump_time - sample_period, end=jump_time,
            missing_seconds=missing, source=SOURCE_TIMER))
    return sorted(interruptions, key=lambda interruption: interruption.start)


def _locate_with_timer(interruptions: list[RecordingInterruption], jump_time: float,
                       sample_period: float, missing: float) -> None:
    """Interrupção vista pelo GPS num trecho sem posição: se o cronômetro saltou o
    mesmo tanto lá dentro, o salto diz em que amostra a gravação parou. A duração
    continua sendo a do GPS. Sem interrupção do GPS em volta, o salto é ignorado."""
    for position, interruption in enumerate(interruptions):
        inside = interruption.start < jump_time <= interruption.end + sample_period
        if (inside and not interruption.located
                and abs(interruption.missing_seconds - missing) <= TIMER_MATCH_TOLERANCE_SECONDS):
            interruptions[position] = RecordingInterruption(
                start=jump_time - sample_period, end=jump_time,
                missing_seconds=interruption.missing_seconds, source=SOURCE_GPS)
            return


def absolute_time(session: "Session", time: np.ndarray) -> np.ndarray:
    """Horário UTC (datetime64[ms]) de cada tempo do log; NaT quando não dá para saber.

    Cada trecho contínuo de gravação tem a sua âncora: horário = tempo do log +
    deslocamento, com o deslocamento ajustado nas amostras em que "GPS Time" acabou de
    mudar de valor. Amostra sem posição dentro de um trecho contínuo recebe horário
    pela âncora. Amostra dentro de uma interrupção não localizada fica sem horário:
    não dá para saber de que lado da parada ela está.
    """
    time = np.atleast_1d(np.asarray(time, dtype=np.float64))
    result = np.full(len(time), np.nan)
    clock = session.gps_clock
    if clock is None or not len(clock.time):
        return _to_datetime(result)

    interruptions = session.interruptions
    starts = [-np.inf] + [interruption.end for interruption in interruptions]
    ends = [interruption.start for interruption in interruptions] + [np.inf]
    for start, end in zip(starts, ends):
        anchor = _segment_anchor(clock, start, end)
        if anchor is None:
            continue
        selected = (time >= start) & (time <= end)
        result[selected] = time[selected] + anchor(time[selected])
    return _to_datetime(result)


def _segment_anchor(clock: GpsClock, start: float, end: float):
    inside = (clock.time >= start) & (clock.time <= end)
    if not inside.any():
        return None
    # Na amostra em que o valor acabou de mudar, o horário verdadeiro está entre o
    # valor e o valor mais um período de amostragem: o meio do período é a estimativa.
    half_period = clock.sample_period / 2
    ticks = inside & clock.ticked
    if ticks.sum() < 2:
        offset = float(np.median(clock.seconds[inside] - clock.time[inside]))
        return lambda time: offset
    tick_time = clock.time[ticks]
    tick_offset = clock.seconds[ticks] - tick_time + half_period
    base = float(np.median(tick_offset))
    if tick_time[-1] - tick_time[0] < LINEAR_ANCHOR_MINIMUM_SECONDS:
        return lambda time: base
    center = float(tick_time.mean())
    slope, intercept = np.polyfit(tick_time - center, tick_offset - base, 1)
    return lambda time: base + intercept + slope * (time - center)


def _to_datetime(epoch_seconds: np.ndarray) -> np.ndarray:
    milliseconds = np.zeros(len(epoch_seconds), dtype=np.int64)
    known = np.isfinite(epoch_seconds)
    milliseconds[known] = np.round(epoch_seconds[known] * 1000).astype(np.int64)
    result = milliseconds.astype("datetime64[ms]")
    result[~known] = np.datetime64("NaT", "ms")
    return result
