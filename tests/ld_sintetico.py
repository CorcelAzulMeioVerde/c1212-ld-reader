"""Gerador de um .ld sintético do painel C1212, para os testes rodarem sem nenhum log real.

Grava o mesmo leiaute que o leitor espera (binary_format.py) com uma sessão inventada de
400 s (tempo do log), escolhida para exercitar todas as regras do leitor:

  - GPS sem posição nos primeiros 5 s (qualidade 0, latitude e longitude zeradas);
  - perda de dados de 100,0 s a 100,5 s: todos os canais rápidos em -1, e a ECU reinicia;
  - -1 legítimo em "G Force Lat" (-0,01 G) fora da perda;
  - passagens pela linha em 50, 150, 250 e 370 s de tempo real;
  - 60 s sem gravação a partir de 300 s do tempo do log: o painel parou e voltou, e o arquivo
    não marca a interrupção. Só o relógio do GPS (e o salto do cronômetro) mostram.

Voltas esperadas: 0 saída, 1 e 2 cronometradas (100,00 s), 3 interrompida (oficial 120,00 s,
medida 60 s), 4 incompleta.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motec_dash import binary_format as bf

DURATION = 400.0                       # segundos de tempo do log
INTERRUPTION_AT = 300.0                # tempo do log em que a gravação parou
INTERRUPTION_SECONDS = 60.0            # tempo real sem gravação
CROSSINGS_REAL = (50.0, 150.0, 250.0, 370.0)
DROPOUT = (100.0, 100.5)
NO_FIX_UNTIL = 5.0
LEGITIMATE_MINUS_ONE_EVERY = 1000      # amostras de "G Force Lat"
START_UTC = np.datetime64("2026-01-10T14:00:00", "s")
DEVICE_SERIAL = 12345

HEADER = {"driver": "Piloto Teste", "vehicle": "Carro Teste", "engine": "Motor 1",
          "venue": "Autódromo Teste", "team": "Equipe Teste", "short_comment": "curto",
          "event": "Evento Teste", "session": "Treino", "long_comment": "Engine: 1",
          "vehicle_identification": "Veículo 7"}

# Leiaute dos blocos (os ponteiros do local e do veículo têm 2 bytes).
EVENT_BLOCK = 0x0700
VENUE_BLOCK = 0x0C00
VEHICLE_BLOCK = 0x1100
FIRST_CHANNEL_HEADER = 0x1200


@dataclass(frozen=True)
class SyntheticChannel:
    name: str
    short_name: str
    unit: str
    frequency: int
    type_code: tuple[int, int]
    decimal_places: int
    raw: np.ndarray


def log_time(frequency: int) -> np.ndarray:
    return np.arange(int(DURATION * frequency)) / frequency


def real_time(time: np.ndarray) -> np.ndarray:
    return np.where(time >= INTERRUPTION_AT, time + INTERRUPTION_SECONDS, time)


def _crossings_before(real: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(CROSSINGS_REAL), real, side="right")


def _in_dropout(time: np.ndarray) -> np.ndarray:
    return (time >= DROPOUT[0]) & (time < DROPOUT[1])


def _to_raw(values: np.ndarray, decimal_places: int) -> np.ndarray:
    return np.round(values * 10 ** decimal_places).astype(np.int64)


def build_channels() -> list[SyntheticChannel]:
    channels: list[SyntheticChannel] = []

    def add(name, short, unit, frequency, type_code, decimal_places, raw, dropout=True):
        raw = np.asarray(raw, dtype=np.int64).copy()
        if dropout and frequency >= 10:
            raw[_in_dropout(log_time(frequency))] = -1
        channels.append(SyntheticChannel(name, short, unit, frequency, type_code, decimal_places, raw))

    int16, int32 = (3, 2), (5, 4)

    # Cronômetro da volta: conta o tempo real desde a última passagem (0,01 s).
    time = log_time(100)
    real = real_time(time)
    last = np.concatenate([[0.0], CROSSINGS_REAL])[_crossings_before(real)]
    add("Running Lap Time", "RLapT", "s", 100, int16, 2, _to_raw(real - last, 2))

    # Tempo oficial da última volta completa (5 Hz) e número da volta (1 Hz).
    time = log_time(5)
    completed = _crossings_before(real_time(time))
    official = np.diff(np.concatenate([[0.0], CROSSINGS_REAL]))
    lap_time = np.where(completed > 0, official[np.maximum(completed - 1, 0)], 0.0)
    add("Lap Time", "LapT", "s", 5, int16, 2, _to_raw(lap_time, 2))
    add("Lap Number", "LapN", "", 1, int16, 0, _crossings_before(real_time(log_time(1))))

    # Canais de 100 Hz do carro.
    time = log_time(100)
    add("Throttle Position", "TPS", "%", 100, int16, 1,
        _to_raw(50 + 40 * np.sin(time / 7), 1))
    add("Engine Speed", "RPM", "rpm", 100, int16, 0,
        np.round(4000 + 2000 * np.sin(time / 5)))
    g_force = _to_raw(1.2 * np.sin(time / 3), 2)
    g_force[::LEGITIMATE_MINUS_ONE_EVERY] = -1          # -0,01 G legítimo
    add("G Force Lat", "GLat", "G", 100, int16, 2, g_force)

    # Velocidade (20 Hz) e tempo ligado da ECU (10 Hz), que zera depois da perda.
    time = log_time(20)
    add("Ground Speed", "GSpd", "km/h", 20, int16, 1, _to_raw(120 + 60 * np.sin(time / 9), 1))
    time = log_time(10)
    uptime = np.where(time >= DROPOUT[1], time - DROPOUT[1], time + 1000)
    add("ECU Uptime", "Uptm", "s", 10, int32, 1, _to_raw(uptime, 1))

    # GPS a 20 Hz: relógio UTC (hhmmss com 0,1 s em milésimos; data ddmmaa 1024 semanas
    # atrasada, como sai na maioria dos painéis), qualidade e posição.
    time = log_time(20)
    no_fix = time < NO_FIX_UNTIL
    tenths = np.floor(real_time(time) * 10 + 1e-6).astype(np.int64)
    moment = START_UTC + (tenths // 10).astype("timedelta64[s]")
    seconds_of_day = (moment - moment.astype("datetime64[D]")).astype(np.int64)
    hhmmss = (seconds_of_day // 3600) * 10000 + (seconds_of_day // 60 % 60) * 100 + seconds_of_day % 60
    gps_time = hhmmss * 1000 + (tenths % 10) * 100
    gps_time[no_fix] = 123                              # lixo: não é múltiplo de 100
    date = (moment.astype("datetime64[D]") - np.timedelta64(1024 * 7, "D")).astype(object)
    gps_date = np.array([d.day * 10000 + d.month * 100 + d.year % 100 for d in date])
    add("GPS Time", "GTime", "", 20, int32, 0, gps_time)
    add("GPS Date", "GDate", "", 20, int32, 0, gps_date)
    add("GPS Quality", "GQual", "", 20, int16, 0, np.where(no_fix, 0, 2))
    angle = real_time(time) / 100 * 2 * np.pi
    latitude = np.where(no_fix, 0, _to_raw(-20.0 + 0.004 * np.sin(angle), 7))
    longitude = np.where(no_fix, 0, _to_raw(-45.0 + 0.004 * np.cos(angle), 7))
    add("GPS Latitude", "GLat", "deg", 20, int32, 7, latitude)
    add("GPS Longitude", "GLong", "deg", 20, int32, 7, longitude)
    gps_speed = _to_raw(120 + 60 * np.sin(time / 9), 1)
    gps_speed[no_fix] = 9999
    add("GPS Speed", "GSpd", "km/h", 20, int16, 1, gps_speed)
    return channels


def _text(value: str, size: int) -> bytes:
    return value.encode("latin-1").ljust(size, b"\0")[:size]


def write_ld(path: Path, device_type: str = "C1212") -> Path:
    channels = build_channels()
    data_start = FIRST_CHANNEL_HEADER + len(channels) * bf.CHANNEL_HEADER_SIZE
    blocks = [np.asarray(channel.raw, dtype=bf.DATA_TYPES[channel.type_code]).tobytes()
              for channel in channels]
    content = bytearray(data_start + sum(len(block) for block in blocks))

    struct.pack_into("<I", content, 0, bf.HEADER_MARKER)
    struct.pack_into("<I", content, bf.OFFSET_CHANNEL_METADATA_POINTER, FIRST_CHANNEL_HEADER)
    struct.pack_into("<I", content, bf.OFFSET_CHANNEL_DATA_POINTER, data_start)
    struct.pack_into("<I", content, bf.OFFSET_EVENT_POINTER, EVENT_BLOCK)
    struct.pack_into("<I", content, bf.OFFSET_DEVICE_SERIAL, DEVICE_SERIAL)
    content[bf.OFFSET_DEVICE_TYPE:bf.OFFSET_DEVICE_TYPE + 8] = _text(device_type, 8)
    struct.pack_into("<H", content, bf.OFFSET_DEVICE_VERSION, 100)
    struct.pack_into("<I", content, bf.OFFSET_CHANNEL_COUNT, len(channels))
    content[bf.OFFSET_DATE:bf.OFFSET_DATE + 16] = _text("10/01/2026", 16)
    content[bf.OFFSET_TIME:bf.OFFSET_TIME + 16] = _text("11:00:00", 16)
    for offset, key in ((bf.OFFSET_DRIVER, "driver"), (bf.OFFSET_VEHICLE, "vehicle"),
                        (bf.OFFSET_ENGINE, "engine"), (bf.OFFSET_VENUE, "venue"),
                        (bf.OFFSET_SHORT_COMMENT, "short_comment"), (bf.OFFSET_TEAM, "team")):
        content[offset:offset + 64] = _text(HEADER[key], 64)

    content[EVENT_BLOCK:EVENT_BLOCK + 64] = _text(HEADER["event"], 64)
    content[EVENT_BLOCK + 64:EVENT_BLOCK + 128] = _text(HEADER["session"], 64)
    content[EVENT_BLOCK + 128:EVENT_BLOCK + 1152] = _text(HEADER["long_comment"], 1024)
    struct.pack_into("<H", content, EVENT_BLOCK + bf.OFFSET_VENUE_POINTER_IN_EVENT_BLOCK, VENUE_BLOCK)
    content[VENUE_BLOCK:VENUE_BLOCK + 64] = _text(HEADER["venue"], 64)
    struct.pack_into("<H", content, VENUE_BLOCK + bf.OFFSET_VEHICLE_POINTER_IN_VENUE_BLOCK, VEHICLE_BLOCK)
    content[VEHICLE_BLOCK:VEHICLE_BLOCK + 64] = _text(HEADER["vehicle_identification"], 64)

    data_offset = data_start
    for index, (channel, block) in enumerate(zip(channels, blocks)):
        pointer = FIRST_CHANNEL_HEADER + index * bf.CHANNEL_HEADER_SIZE
        previous = pointer - bf.CHANNEL_HEADER_SIZE if index else 0
        following = pointer + bf.CHANNEL_HEADER_SIZE if index < len(channels) - 1 else 0
        struct.pack_into(bf.CHANNEL_HEADER_FORMAT, content, pointer,
                         previous, following, data_offset, len(channel.raw), 0,
                         *channel.type_code, channel.frequency, 0, 1, 1, channel.decimal_places,
                         _text(channel.name, 32), _text(channel.short_name, 8), _text(channel.unit, 12))
        content[data_offset:data_offset + len(block)] = block
        data_offset += len(block)

    path.write_bytes(bytes(content))
    return path
