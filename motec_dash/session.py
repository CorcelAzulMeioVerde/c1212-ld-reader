"""Sessão de log do painel: metadados, acesso a canais e janelas de perda de dados.

Valor bruto -1 NÃO é tratado como inválido de forma global. Nos logs reais ele é
um valor legítimo em vários canais (por exemplo -0,01 G em aceleração lateral).
Ele só significa "sem dado" dentro de janelas em que a maior parte dos canais
fica em -1 ao mesmo tempo (perda de comunicação). Só essas amostras viram NaN.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import binary_format as bf
from .laps import Lap, segment_laps
from .recording import (GpsClock, RecordingInterruption, absolute_time, find_interruptions,
                        read_gps_clock)

NO_DATA_RAW_VALUE = -1
DROPOUT_GRID_FREQUENCY = 10          # Hz da grade usada para detectar perdas
DROPOUT_MINIMUM_FRACTION = 0.5       # fração de canais simultâneos em -1
# Perda mais curta que o intervalo dos canais lentos só aparece nos rápidos: a maioria
# é medida também dentro de cada frequência de amostragem com canais suficientes.
DROPOUT_MINIMUM_GROUP_CHANNELS = 10
DROPOUT_MARGIN_SECONDS = 0.2
# "ECU Uptime" é conferido até este tanto depois da perda. Se a gravação foi
# interrompida nesse trecho, o valor de depois não diz nada sobre a janela.
DROPOUT_RESTART_SEARCH_SECONDS = 3.0

# Canais que só valem com posição do GPS resolvida. Sem posição, o receptor grava
# latitude e longitude zeradas e a velocidade fica com lixo (visto após reinício da ECU).
GPS_FIX_DEPENDENT_CHANNELS = frozenset({
    "GPS Latitude", "GPS Longitude", "GPS Altitude", "GPS Heading", "GPS Speed"})
GPS_NO_FIX_QUALITY = 0


@dataclass(frozen=True)
class DropoutWindow:
    start: float
    end: float
    peak_fraction: float
    ecu_restarted: bool | None = None           # "ECU Uptime" zerou nesta janela
    ground_speed_before: float | None = None    # km/h, 1 s antes da janela


@dataclass
class ChannelData:
    info: bf.ChannelInfo
    time: np.ndarray
    values: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float(np.isfinite(self.values).mean()) if len(self.values) else 0.0


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._buffer = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.header = bf.parse_header(self._buffer)
        channel_list = bf.parse_channels(self._buffer)
        self.structure = bf.validate_structure(self._buffer, self.header, channel_list)
        self.channels: dict[str, bf.ChannelInfo] = {}
        for channel in channel_list:
            # Nome repetido: mantém o primeiro e registra aviso (já em structure.warnings).
            self.channels.setdefault(channel.name, channel)
        self._dropout_windows: list[DropoutWindow] | None = None
        self._dropouts: list[DropoutWindow] | None = None
        self._gps_clock: GpsClock | None = None
        self._gps_clock_read = False
        self._interruptions: list[RecordingInterruption] | None = None
        self._laps: list[Lap] | None = None

    # ------------------------------------------------------------------ básicos
    @property
    def duration(self) -> float:
        return max(c.duration for c in self.channels.values())

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with open(self.path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def raw(self, name: str) -> np.ndarray:
        return bf.read_raw(self._buffer, self._info(name))

    def channel(self, name: str, mask_dropouts: bool = True, first_sample: int = 0,
                last_sample: int | None = None) -> ChannelData:
        """Canal inteiro ou, com first_sample e last_sample (exclusivo), só um trecho:
        os valores do trecho são os mesmos do canal inteiro."""
        info = self._info(name)
        selected = slice(first_sample, info.sample_count if last_sample is None
                         else min(last_sample, info.sample_count))
        raw = bf.read_raw(self._buffer, info)[selected]
        values = bf.scale_values(raw, info)
        time = np.arange(selected.start, max(selected.stop, selected.start)) / info.frequency
        if mask_dropouts:
            invalid = (raw == NO_DATA_RAW_VALUE) & self._inside_dropout(time)
            values[invalid] = np.nan
            if name in GPS_FIX_DEPENDENT_CHANNELS:
                no_fix = self._gps_without_fix()
                if no_fix is not None and len(no_fix) == info.sample_count:
                    values[no_fix[selected]] = np.nan
        return ChannelData(info=info, time=time, values=values)

    def _gps_without_fix(self) -> np.ndarray | None:
        needed = {"GPS Quality", "GPS Latitude", "GPS Longitude"}
        if not needed <= self.channels.keys():
            return None
        quality = bf.scale_values(self.raw("GPS Quality"), self.channels["GPS Quality"])
        latitude = self.raw("GPS Latitude")
        longitude = self.raw("GPS Longitude")
        if not len(quality) == len(latitude) == len(longitude):
            return None
        return (quality == GPS_NO_FIX_QUALITY) | ((latitude == 0) & (longitude == 0))

    def _info(self, name: str) -> bf.ChannelInfo:
        try:
            return self.channels[name]
        except KeyError:
            raise KeyError(f"Canal '{name}' não existe neste log.") from None

    # ------------------------------------------------------- perda de dados
    @property
    def dropouts(self) -> list[DropoutWindow]:
        if self._dropouts is None:
            self._dropouts = [self._classify_dropout(window)
                              for window in self._unclassified_dropouts()]
        return self._dropouts

    def _unclassified_dropouts(self) -> list[DropoutWindow]:
        """As janelas sem classificação bastam para marcar dado ausente; a classificação
        depende das interrupções de gravação, que por sua vez leem canais."""
        if self._dropout_windows is None:
            self._dropout_windows = self._detect_dropouts()
        return self._dropout_windows

    def _classify_dropout(self, window: DropoutWindow) -> DropoutWindow:
        """Nos logs reais, as perdas coincidem com reinício da ECU. Registra isso e a
        velocidade antes da perda: reinício com o carro andando é evento grave."""
        restarted = speed = None
        if "ECU Uptime" in self.channels:
            uptime = self.channel("ECU Uptime", mask_dropouts=False)
            before = uptime.values[(uptime.time >= window.start - 1.5)
                                   & (uptime.time < window.start) & (uptime.values >= 0)]
            after = uptime.values[(uptime.time > window.end)
                                  & (uptime.time <= window.end + DROPOUT_RESTART_SEARCH_SECONDS)
                                  & (uptime.values >= 0)]
            if len(before) and len(after):
                restarted = bool(after.min() < before.max())
            # Perda seguida de interrupção de gravação: "ECU Uptime" menor depois de um
            # intervalo sem gravação não prova que o reinício foi na janela.
            if any(window.start - DROPOUT_MARGIN_SECONDS <= interruption.start
                   <= window.end + DROPOUT_RESTART_SEARCH_SECONDS
                   for interruption in self.interruptions):
                restarted = None
        if "Ground Speed" in self.channels:
            ground = self.channel("Ground Speed", mask_dropouts=False)
            sample = ground.values[(ground.time >= window.start - 1.0)
                                   & (ground.time < window.start - 0.5) & (ground.values >= 0)]
            speed = float(sample.max()) if len(sample) else None
        return DropoutWindow(window.start, window.end, window.peak_fraction, restarted, speed)

    def _detect_dropouts(self) -> list[DropoutWindow]:
        bins = int(np.floor(self.duration * DROPOUT_GRID_FREQUENCY))
        hits_by_frequency: dict[int, np.ndarray] = {}
        contributors_by_frequency: dict[int, int] = {}
        for info in self.channels.values():
            if info.frequency < DROPOUT_GRID_FREQUENCY or info.frequency % DROPOUT_GRID_FREQUENCY:
                continue
            per_bin = info.frequency // DROPOUT_GRID_FREQUENCY
            usable = min(bins, info.sample_count // per_bin)
            raw = bf.read_raw(self._buffer, info)[: usable * per_bin].reshape(usable, per_bin)
            hits = hits_by_frequency.setdefault(info.frequency, np.zeros(bins))
            hits[:usable] += (raw == NO_DATA_RAW_VALUE).any(axis=1)
            contributors_by_frequency[info.frequency] = (
                contributors_by_frequency.get(info.frequency, 0) + 1)
        if not contributors_by_frequency:
            return []
        # Maior fração entre o conjunto de todos os canais e cada frequência de amostragem.
        fraction = sum(hits_by_frequency.values()) / sum(contributors_by_frequency.values())
        for frequency, contributors in contributors_by_frequency.items():
            if contributors >= DROPOUT_MINIMUM_GROUP_CHANNELS:
                fraction = np.maximum(fraction, hits_by_frequency[frequency] / contributors)
        flagged = np.flatnonzero(fraction >= DROPOUT_MINIMUM_FRACTION)
        windows: list[DropoutWindow] = []
        for index in flagged:
            start = index / DROPOUT_GRID_FREQUENCY
            end = (index + 1) / DROPOUT_GRID_FREQUENCY
            if windows and start - windows[-1].end <= 1 / DROPOUT_GRID_FREQUENCY:
                previous = windows[-1]
                windows[-1] = DropoutWindow(previous.start, end,
                                            max(previous.peak_fraction, float(fraction[index])))
            else:
                windows.append(DropoutWindow(start, end, float(fraction[index])))
        return windows

    def _inside_dropout(self, time: np.ndarray) -> np.ndarray:
        inside = np.zeros(len(time), dtype=bool)
        for window in self._unclassified_dropouts():
            inside |= ((time >= window.start - DROPOUT_MARGIN_SECONDS)
                       & (time <= window.end + DROPOUT_MARGIN_SECONDS))
        return inside

    # ------------------------------------------- interrupções e horário
    @property
    def gps_clock(self) -> GpsClock | None:
        if not self._gps_clock_read:
            self._gps_clock = read_gps_clock(self)
            self._gps_clock_read = True
        return self._gps_clock

    @property
    def interruptions(self) -> list[RecordingInterruption]:
        if self._interruptions is None:
            self._interruptions = find_interruptions(self)
        return self._interruptions

    @property
    def start_time(self) -> datetime.datetime | None:
        """Primeiro horário válido do GPS, em UTC. O cabeçalho não entra: a data e o
        horário dele vêm do relógio do painel, que não é confiável."""
        clock = self.gps_clock
        if clock is None or not len(clock.seconds):
            return None
        return datetime.datetime.fromtimestamp(float(clock.seconds[0]), datetime.timezone.utc)

    def absolute_time(self, time: np.ndarray) -> np.ndarray:
        """Horário UTC (datetime64[ms]) de cada tempo do log; NaT quando não dá para saber."""
        return absolute_time(self, time)

    # ---------------------------------------------------------------- voltas
    @property
    def laps(self) -> list[Lap]:
        if self._laps is None:
            self._laps = segment_laps(
                running_lap_time=self.channel("Running Lap Time"),
                lap_time=self.channel("Lap Time"),
                lap_number=self.channel("Lap Number"),
                session_duration=self.duration,
                interruptions=self.interruptions,
            )
        return self._laps

    def timed_laps(self) -> list[Lap]:
        return [lap for lap in self.laps if lap.kind == "timed"]

    def fastest_lap(self) -> Lap | None:
        timed = self.timed_laps()
        return min(timed, key=lambda lap: lap.official_time) if timed else None

    def close(self) -> None:
        del self._buffer

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
