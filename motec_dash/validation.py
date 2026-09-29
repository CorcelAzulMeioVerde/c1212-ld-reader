"""Validação de um arquivo antes de ele ser aceito como log do painel.

O que não é .ld, o que é .ld de outro dispositivo (por exemplo, a ECU M1) e o que
tem estrutura inválida é recusado, com uma mensagem que explica o motivo. O dispositivo vem do campo de tipo do cabeçalho, que é gravado pelo
equipamento; os campos de texto preenchidos pelas equipes nunca são usados.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .binary_format import HEADER_MARKER, HEADER_MINIMUM_SIZE, LdFormatError, SessionHeader

DASH_DEVICE_TYPE = "C1212"


class NotAnLdFileError(LdFormatError):
    """O arquivo não é um .ld da MoTeC."""


class NotADashLogError(LdFormatError):
    """É um .ld, mas de outro dispositivo (por exemplo, a ECU M1)."""


class InvalidStructureError(LdFormatError):
    """É um .ld do painel, mas a estrutura está inválida (truncado ou corrompido)."""


def check_dash_log(path: str | Path) -> SessionHeader:
    """Devolve o cabeçalho se o arquivo for um log íntegro do painel C1212."""
    from .session import Session

    path = Path(path)
    with open(path, "rb") as file:
        beginning = file.read(4)
    if len(beginning) < 4 or path.stat().st_size < HEADER_MINIMUM_SIZE \
            or struct.unpack("<I", beginning)[0] != HEADER_MARKER:
        raise NotAnLdFileError("O arquivo não é um log .ld da MoTeC.")

    session = None
    try:
        session = Session(path)
        header = session.header
    except LdFormatError as error:
        device_type = _device_type(path)
        if device_type is not None and device_type != DASH_DEVICE_TYPE:
            raise _not_a_dash_log(device_type) from error
        raise InvalidStructureError(f"Log do painel com estrutura inválida: {error}") from error
    finally:
        if session is not None:
            session.close()
    if header.device_type != DASH_DEVICE_TYPE:
        raise _not_a_dash_log(header.device_type)
    return header


def _not_a_dash_log(device_type: str) -> NotADashLogError:
    return NotADashLogError(f"O arquivo é um log de {device_type or 'dispositivo desconhecido'}, não do "
                            f"painel {DASH_DEVICE_TYPE}. Só o log do painel é aceito.")


def _device_type(path: Path) -> str | None:
    from .binary_format import OFFSET_DEVICE_TYPE, _decode_text

    with open(path, "rb") as file:
        file.seek(OFFSET_DEVICE_TYPE)
        raw = file.read(8)
    return _decode_text(raw) if len(raw) == 8 else None
