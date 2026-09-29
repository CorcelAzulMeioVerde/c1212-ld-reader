"""Leitura de baixo nível do formato binário .ld da MoTeC (logger de painel C1212).

O formato não tem documentação pública. Os deslocamentos abaixo vêm de engenharia
reversa e foram validados byte a byte em logs reais: os blocos de
dados de todos os canais ficam contíguos e terminam exatamente no fim do arquivo.

Esta camada só entrega estrutura e valores brutos já escalados. Regras de sessão
(perda de dados, voltas) ficam em session.py e laps.py.
"""

from __future__ import annotations

import datetime
import struct
from dataclasses import dataclass, field

import numpy as np

HEADER_MARKER = 0x40

# Deslocamentos do cabeçalho principal.
OFFSET_CHANNEL_METADATA_POINTER = 0x08
OFFSET_CHANNEL_DATA_POINTER = 0x0C
OFFSET_EVENT_POINTER = 0x24
OFFSET_DEVICE_SERIAL = 0x46
OFFSET_DEVICE_TYPE = 0x4A
OFFSET_DEVICE_VERSION = 0x52
OFFSET_CHANNEL_COUNT = 0x56
OFFSET_DATE = 0x5E
OFFSET_TIME = 0x7E
OFFSET_DRIVER = 0x9E
OFFSET_VEHICLE = 0xDE
OFFSET_ENGINE = 0x11E
OFFSET_VENUE = 0x15E
OFFSET_SHORT_COMMENT = 0x624
OFFSET_TEAM = 0x694
HEADER_MINIMUM_SIZE = OFFSET_TEAM + 64

# Bloco de evento: nome (64), sessão (64), comentário longo (1024) e o ponteiro do bloco do local.
EVENT_BLOCK_SIZE = 64 + 64 + 1024 + 2
OFFSET_VENUE_POINTER_IN_EVENT_BLOCK = 1152
# Bloco do local: nome (64), e no fim o ponteiro do bloco do veículo.
OFFSET_VEHICLE_POINTER_IN_VENUE_BLOCK = 1098
VENUE_BLOCK_SIZE = OFFSET_VEHICLE_POINTER_IN_VENUE_BLOCK + 2
# Bloco do veículo: o primeiro campo é a identificação do veículo (64).
VEHICLE_IDENTIFICATION_SIZE = 64

# Cabeçalho de canal: 124 bytes.
CHANNEL_HEADER_FORMAT = "<IIIIHHHHhhhh32s8s12s40x"
CHANNEL_HEADER_SIZE = struct.calcsize(CHANNEL_HEADER_FORMAT)

# (código de tipo A, tamanho em bytes) -> tipo numpy.
# Só os três primeiros aparecem no logger de painel; os demais existem no formato
# (vistos no log da ECU M1) e ficam suportados para não falhar em silêncio.
DATA_TYPES: dict[tuple[int, int], str] = {
    (0, 2): "<i2",
    (3, 2): "<i2",
    (5, 4): "<i4",
    (6, 4): "<u4",
    (7, 2): "<f2",
    (7, 4): "<f4",
    (8, 8): "<f8",
}


class LdFormatError(Exception):
    """Arquivo estruturalmente inválido: não deve ser processado."""


class UnsupportedDataTypeError(LdFormatError):
    """Canal com código de tipo de dado desconhecido."""


def _decode_text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1").strip()


def _decode_header_text(raw: bytes) -> str:
    """Texto escrito pela equipe no cabeçalho, como veio: só sai o preenchimento depois do
    primeiro NUL. Nada é aparado, porque esses campos são guardados para conferência."""
    return raw.split(b"\0", 1)[0].decode("latin-1")


@dataclass(frozen=True)
class SessionHeader:
    """Os textos (piloto, veículo, motor, local, equipe, identificação do veículo, evento, sessão
    e comentários) são preenchidos pelas equipes, sem padrão: servem para uma pessoa ler e para
    conferir depois, nunca para identificar carro, evento, sessão ou autódromo."""
    device_type: str
    device_serial: int
    device_version: int
    declared_channel_count: int
    start: datetime.datetime | None
    driver: str
    vehicle: str
    engine: str
    venue: str
    team: str
    # None: o log não tem bloco do veículo (é diferente de campo vazio).
    vehicle_identification: str | None
    event: str
    session: str
    short_comment: str
    long_comment: str


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    name: str
    short_name: str
    unit: str
    frequency: int
    sample_count: int
    numpy_type: str
    type_code: tuple[int, int]
    shift: int
    multiplier: int
    scale: int
    decimal_places: int
    header_offset: int
    data_offset: int

    @property
    def byte_length(self) -> int:
        return self.sample_count * np.dtype(self.numpy_type).itemsize

    @property
    def duration(self) -> float:
        return self.sample_count / self.frequency


@dataclass
class StructureReport:
    file_size: int
    channel_count: int
    data_start: int
    data_end: int
    gaps: list[tuple[int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def data_ends_at_file_end(self) -> bool:
        return self.data_end == self.file_size


def parse_header(buffer: memoryview | bytes) -> SessionHeader:
    if len(buffer) < HEADER_MINIMUM_SIZE:
        raise LdFormatError("Arquivo menor que o cabeçalho mínimo.")
    marker, = struct.unpack_from("<I", buffer, 0)
    if marker != HEADER_MARKER:
        raise LdFormatError(f"Marcador de cabeçalho inesperado: {marker:#x}.")

    date_text = _decode_text(bytes(buffer[OFFSET_DATE:OFFSET_DATE + 16]))
    time_text = _decode_text(bytes(buffer[OFFSET_TIME:OFFSET_TIME + 16]))
    start = None
    for pattern in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            start = datetime.datetime.strptime(f"{date_text} {time_text}", pattern)
            break
        except ValueError:
            continue

    event_pointer, = struct.unpack_from("<I", buffer, OFFSET_EVENT_POINTER)
    event = session = long_comment = ""
    vehicle_identification = None
    if event_pointer:
        if event_pointer + EVENT_BLOCK_SIZE > len(buffer):
            raise LdFormatError("Ponteiro do bloco de evento fora do arquivo.")
        block = bytes(buffer[event_pointer:event_pointer + EVENT_BLOCK_SIZE])
        event = _decode_header_text(block[0:64])
        session = _decode_header_text(block[64:128])
        long_comment = _decode_header_text(block[128:1152])
        venue_pointer, = struct.unpack_from("<H", block, OFFSET_VENUE_POINTER_IN_EVENT_BLOCK)
        vehicle_identification = _vehicle_identification(buffer, venue_pointer)

    def header_text(offset: int) -> str:
        return _decode_header_text(bytes(buffer[offset:offset + 64]))

    return SessionHeader(
        device_type=_decode_text(bytes(buffer[OFFSET_DEVICE_TYPE:OFFSET_DEVICE_TYPE + 8])),
        device_serial=struct.unpack_from("<I", buffer, OFFSET_DEVICE_SERIAL)[0],
        device_version=struct.unpack_from("<H", buffer, OFFSET_DEVICE_VERSION)[0],
        declared_channel_count=struct.unpack_from("<I", buffer, OFFSET_CHANNEL_COUNT)[0],
        start=start,
        driver=header_text(OFFSET_DRIVER),
        vehicle=header_text(OFFSET_VEHICLE),
        engine=header_text(OFFSET_ENGINE),
        venue=header_text(OFFSET_VENUE),
        team=header_text(OFFSET_TEAM),
        vehicle_identification=vehicle_identification,
        event=event,
        session=session,
        short_comment=header_text(OFFSET_SHORT_COMMENT),
        long_comment=long_comment,
    )


def _vehicle_identification(buffer: memoryview | bytes, venue_pointer: int) -> str | None:
    """Cadeia de ponteiros evento -> local -> veículo. Ponteiro zerado ou fora do arquivo é log
    sem bloco do veículo: o campo é só informativo, e o log não é recusado por causa dele."""
    if not venue_pointer or venue_pointer + VENUE_BLOCK_SIZE > len(buffer):
        return None
    vehicle_pointer, = struct.unpack_from("<H", buffer, venue_pointer + OFFSET_VEHICLE_POINTER_IN_VENUE_BLOCK)
    if not vehicle_pointer or vehicle_pointer + VEHICLE_IDENTIFICATION_SIZE > len(buffer):
        return None
    return _decode_header_text(bytes(buffer[vehicle_pointer:vehicle_pointer + VEHICLE_IDENTIFICATION_SIZE]))


def parse_channels(buffer: memoryview | bytes) -> list[ChannelInfo]:
    """Percorre a lista encadeada de cabeçalhos de canal, com proteção contra ciclos."""
    file_size = len(buffer)
    pointer, = struct.unpack_from("<I", buffer, OFFSET_CHANNEL_METADATA_POINTER)
    channels: list[ChannelInfo] = []
    visited: set[int] = set()
    previous_pointer = 0

    while pointer:
        if pointer in visited:
            raise LdFormatError(f"Ciclo na lista de canais no deslocamento {pointer:#x}.")
        if pointer + CHANNEL_HEADER_SIZE > file_size:
            raise LdFormatError(f"Cabeçalho de canal fora do arquivo: {pointer:#x}.")
        visited.add(pointer)

        (declared_previous, next_pointer, data_offset, sample_count, _counter,
         type_a, type_size, frequency, shift, multiplier, scale, decimal_places,
         name, short_name, unit) = struct.unpack_from(CHANNEL_HEADER_FORMAT, buffer, pointer)
        name = _decode_text(name)

        if declared_previous != previous_pointer:
            raise LdFormatError(f"Encadeamento inconsistente no canal '{name}'.")
        type_code = (type_a, type_size)
        if type_code not in DATA_TYPES:
            raise UnsupportedDataTypeError(f"Canal '{name}': tipo de dado desconhecido {type_code}.")
        if frequency == 0:
            raise LdFormatError(f"Canal '{name}': frequência zero.")
        if scale == 0:
            raise LdFormatError(f"Canal '{name}': fator de escala zero.")

        channel = ChannelInfo(
            index=len(channels), name=name, short_name=_decode_text(short_name),
            unit=_decode_text(unit), frequency=frequency, sample_count=sample_count,
            numpy_type=DATA_TYPES[type_code], type_code=type_code, shift=shift,
            multiplier=multiplier, scale=scale, decimal_places=decimal_places,
            header_offset=pointer, data_offset=data_offset,
        )
        if channel.data_offset + channel.byte_length > file_size:
            raise LdFormatError(f"Canal '{name}': bloco de dados ultrapassa o fim do arquivo.")
        channels.append(channel)
        previous_pointer = pointer
        pointer = next_pointer

    if not channels:
        raise LdFormatError("Nenhum canal encontrado.")
    return channels


def validate_structure(buffer: memoryview | bytes, header: SessionHeader,
                       channels: list[ChannelInfo]) -> StructureReport:
    """Confere se os blocos de dados não se sobrepõem e cobrem a área de dados."""
    blocks = sorted((c.data_offset, c.data_offset + c.byte_length, c.name) for c in channels)
    report = StructureReport(
        file_size=len(buffer), channel_count=len(channels),
        data_start=blocks[0][0], data_end=max(end for _, end, _ in blocks),
    )
    for (start_a, end_a, name_a), (start_b, _, name_b) in zip(blocks, blocks[1:]):
        if end_a > start_b:
            raise LdFormatError(f"Blocos de dados sobrepostos: '{name_a}' e '{name_b}'.")
        if end_a < start_b:
            report.gaps.append((end_a, start_b))

    names = [c.name for c in channels]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        report.warnings.append(f"Nomes de canal repetidos: {duplicated}.")
    if header.declared_channel_count != len(channels):
        report.warnings.append(
            f"Cabeçalho declara {header.declared_channel_count} canais, "
            f"lista encadeada tem {len(channels)}.")
    if report.gaps:
        report.warnings.append(f"{len(report.gaps)} lacunas entre blocos de dados.")
    if not report.data_ends_at_file_end:
        report.warnings.append("Área de dados não termina no fim do arquivo.")
    return report


def read_raw(buffer: np.ndarray, channel: ChannelInfo) -> np.ndarray:
    """Valores brutos (inteiros ou ponto flutuante) exatamente como gravados."""
    return np.frombuffer(buffer, dtype=channel.numpy_type,
                         count=channel.sample_count, offset=channel.data_offset)


def scale_values(raw: np.ndarray, channel: ChannelInfo) -> np.ndarray:
    """Converte para unidade física: (bruto / escala * 10^-casas + deslocamento) * multiplicador."""
    return ((raw.astype(np.float64) / channel.scale) * 10.0 ** (-channel.decimal_places)
            + channel.shift) * channel.multiplier
