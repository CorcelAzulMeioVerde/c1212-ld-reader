"""Segmentação de voltas a partir dos canais do painel (sem usar o .ldx).

Fontes, em ordem de autoridade:
  - "Lap Number" (1 Hz): define QUANTAS passagens houve. Reinício do cronômetro
    sem incremento do número da volta não é passagem (ex.: perda de dados).
  - "Running Lap Time" (100 Hz): define QUANDO, com resolução de 0,01 s. Ele zera
    com um pequeno atraso depois da passagem real.
  - "Lap Time" (5 Hz): tempo oficial da volta calculado pelo painel. Usado para
    descontar o atraso: passagem = instante do zeramento - (cronômetro antes - oficial).

O painel pode parar de gravar e voltar com "Lap Number" zerado. O tempo do log não
registra a interrupção (ver recording.py: ela é achada pelo relógio do GPS, com o
salto do cronômetro como complemento). Por isso o número da volta é a ordem das
passagens (o do painel fica em `dash_number`) e a volta que contém uma interrupção
de gravação é "interrupted" (interrompida), não "timed" (cronometrada).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .recording import RecordingInterruption
    from .session import ChannelData

RESET_MINIMUM_DROP_SECONDS = 5.0
MATCH_WINDOW_BEFORE_SECONDS = 3.0
MATCH_WINDOW_AFTER_SECONDS = 2.0
OFFICIAL_TIME_SEARCH_SECONDS = 2.0
OFFICIAL_TIME_TOLERANCE_SECONDS = 0.5
TIMER_JUMP_MINIMUM_SECONDS = 1.0
# "Lap Number" é amostrado a 1 Hz: a passagem ocorre até 1 s antes da amostra com o novo número.
REFINEMENT_TOLERANCE_SECONDS = 2.0


# Identificadores do tipo de volta. O texto em português existe só para exibição.
LAP_KINDS = ("out", "timed", "interrupted", "incomplete")
LAP_KIND_TEXT = {"out": "saída", "timed": "cronometrada", "interrupted": "interrompida",
                 "incomplete": "incompleta"}


@dataclass(frozen=True)
class Crossing:
    time: float
    official_lap_time: float | None
    precise: bool           # True quando veio do cronômetro de 100 Hz


@dataclass(frozen=True)
class Lap:
    number: int             # ordem das passagens; não depende do contador do painel
    kind: str               # LAP_KINDS: "out", "timed", "interrupted" ou "incomplete"
    start: float
    end: float
    official_time: float | None
    measured_duration: float
    precise: bool
    dash_number: int | None = None      # "Lap Number" do painel; difere se o contador zerou

    @property
    def discrepancy(self) -> float | None:
        if self.official_time is None:
            return None
        return self.measured_duration - self.official_time


def _find_resets(running: "ChannelData") -> list[tuple[float, float]]:
    """(instante do zeramento, valor do cronômetro imediatamente antes)."""
    values = running.values
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return []
    before, after = valid[:-1], valid[1:]
    drop = values[before] - values[after]
    is_reset = (drop > RESET_MINIMUM_DROP_SECONDS) & (drop > 0.5 * values[before])
    return [(float(running.time[a]), float(values[b]))
            for b, a in zip(before[is_reset], after[is_reset])]


def find_timer_jumps(running: "ChannelData") -> list[tuple[float, float]]:
    """(instante, segundos sem gravação): o cronômetro avançou mais que o tempo do log.

    O salto tem de se manter: na borda de uma perda de dados aparece uma amostra
    solta com o bruto 65535 (655,35 s) e o cronômetro volta ao valor de antes."""
    values = running.values
    valid = np.flatnonzero(np.isfinite(values) & (values >= 0))
    if len(valid) < 3:
        return []
    before, after, following = valid[:-2], valid[1:-1], valid[2:]
    excess = (values[after] - values[before]) - (running.time[after] - running.time[before])
    is_jump = (excess > TIMER_JUMP_MINIMUM_SECONDS) & (values[following] >= values[after])
    return [(float(running.time[a]), float(e)) for a, e in zip(after[is_jump], excess[is_jump])]


def _find_lap_increments(lap_number: "ChannelData") -> list[tuple[float, int]]:
    """(instante da amostra com o novo número, novo número)."""
    values = lap_number.values
    valid = np.flatnonzero(np.isfinite(values))
    increments = []
    for previous, current in zip(valid[:-1], valid[1:]):
        if values[current] > values[previous]:
            increments.append((float(lap_number.time[current]), int(values[current])))
    return increments


def _official_time_after(lap_time: "ChannelData", reset_time: float,
                         running_before: float) -> float | None:
    window = ((lap_time.time > reset_time)
              & (lap_time.time <= reset_time + OFFICIAL_TIME_SEARCH_SECONDS)
              & np.isfinite(lap_time.values))
    candidates = lap_time.values[window]
    close = candidates[np.abs(candidates - running_before) <= OFFICIAL_TIME_TOLERANCE_SECONDS]
    return float(close[0]) if len(close) else None


def find_crossings(running_lap_time: "ChannelData", lap_time: "ChannelData",
                   lap_number: "ChannelData") -> list[tuple[Crossing, int]]:
    resets = _find_resets(running_lap_time)
    crossings: list[tuple[Crossing, int]] = []
    used: set[int] = set()
    for increment_time, new_number in _find_lap_increments(lap_number):
        # O número da volta é amostrado a 1 Hz: a mudança ocorreu antes desta amostra.
        candidates = [
            (abs(reset_time - increment_time), index)
            for index, (reset_time, _) in enumerate(resets)
            if index not in used
            and increment_time - MATCH_WINDOW_BEFORE_SECONDS <= reset_time
            <= increment_time + MATCH_WINDOW_AFTER_SECONDS
        ]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            reset_time, running_before = resets[index]
            official = _official_time_after(lap_time, reset_time, running_before)
            delay = (running_before - official) if official is not None else 0.0
            crossings.append((Crossing(reset_time - delay, official, True), new_number))
        else:
            crossings.append((Crossing(increment_time, None, False), new_number))
    return _refine_imprecise_crossings(crossings, running_lap_time)


def _timer_start_before(running: "ChannelData", time: float) -> float | None:
    """Cronômetro parado em zero começa a contar na passagem: passagem = instante - cronômetro."""
    index = min(int(time * running.info.frequency), len(running.values) - 1)
    value = running.values[index]
    if not np.isfinite(value) or not 0 < value <= REFINEMENT_TOLERANCE_SECONDS:
        return None
    return float(running.time[index] - value)


def _refine_imprecise_crossings(crossings: list[tuple[Crossing, int]],
                                running: "ChannelData") -> list[tuple[Crossing, int]]:
    """Passagem sem zeramento do cronômetro (típico da primeira: o cronômetro estava
    parado em zero) tem só a resolução de 1 s do número da volta. Se a passagem
    seguinte é precisa e tem tempo oficial, o início é exato por definição:
    início = fim - tempo oficial. Isso só vale se o resultado cai junto da amostra
    que mostrou o novo número; com um salto do cronômetro no meio da volta não cai,
    e o início vem do próprio cronômetro, que parte de zero na passagem."""
    refined = list(crossings)
    for index in range(len(refined) - 2, -1, -1):
        crossing, number = refined[index]
        following, _ = refined[index + 1]
        if crossing.precise:
            continue
        derived = None
        if following.precise and following.official_lap_time is not None:
            derived = following.time - following.official_lap_time
            if not -REFINEMENT_TOLERANCE_SECONDS <= derived - crossing.time <= 0:
                derived = None
        if derived is None:
            derived = _timer_start_before(running, crossing.time)
        if derived is not None:
            refined[index] = (Crossing(derived, crossing.official_lap_time, True), number)
    return refined


def segment_laps(running_lap_time: "ChannelData", lap_time: "ChannelData",
                 lap_number: "ChannelData", session_duration: float,
                 interruptions: "list[RecordingInterruption]") -> list[Lap]:
    crossings = find_crossings(running_lap_time, lap_time, lap_number)
    if not crossings:
        return [Lap(number=0, kind="incomplete", start=0.0, end=session_duration,
                    official_time=None, measured_duration=session_duration, precise=False)]

    laps: list[Lap] = []
    first, first_number = crossings[0]
    laps.append(Lap(number=first_number - 1, kind="out", start=0.0, end=first.time,
                    official_time=None, measured_duration=first.time, precise=first.precise,
                    dash_number=first_number - 1))

    for order, ((start, dash_number), (end, _)) in enumerate(zip(crossings, crossings[1:])):
        interrupted = any(interruption.start < end.time and interruption.end > start.time
                          for interruption in interruptions)
        laps.append(Lap(number=first_number + order,
                        kind="interrupted" if interrupted else "timed",
                        start=start.time, end=end.time,
                        official_time=end.official_lap_time,
                        measured_duration=end.time - start.time,
                        precise=start.precise and end.precise, dash_number=dash_number))

    last, last_number = crossings[-1]
    laps.append(Lap(number=first_number + len(crossings) - 1, kind="incomplete", start=last.time,
                    end=session_duration, official_time=None,
                    measured_duration=session_duration - last.time, precise=last.precise,
                    dash_number=last_number))
    return laps
