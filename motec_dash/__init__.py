"""Leitor de logs .ld do painel MoTeC C1212 (projeto independente, sem ligação com a MoTeC)."""

from .binary_format import LdFormatError, UnsupportedDataTypeError
from .laps import LAP_KIND_TEXT, LAP_KINDS, Lap
from .recording import INTERRUPTION_SOURCE_TEXT, INTERRUPTION_SOURCES, RecordingInterruption
from .session import ChannelData, DropoutWindow, Session
from .validation import (DASH_DEVICE_TYPE, InvalidStructureError, NotADashLogError,
                         NotAnLdFileError, check_dash_log)

__all__ = ["Session", "LAP_KINDS", "LAP_KIND_TEXT", "ChannelData", "DropoutWindow", "Lap", "RecordingInterruption",
           "INTERRUPTION_SOURCES", "INTERRUPTION_SOURCE_TEXT",
           "LdFormatError", "UnsupportedDataTypeError", "DASH_DEVICE_TYPE", "check_dash_log",
           "NotAnLdFileError", "NotADashLogError", "InvalidStructureError"]
