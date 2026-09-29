"""Linha de comando.

  python -m motec_dash resumo   arquivo.ld [arquivo.ld ...]
  python -m motec_dash exportar arquivo.ld [--saida PASTA]
  python -m motec_dash dicionario [--amostras PASTA] [--saida ARQUIVO]

Sem --saida, a exportação vai para <pasta do .ld>/processado/<nome do arquivo>.
O dicionário lê os logs de --amostras (padrão: MOTEC_AMOSTRAS) e atualiza
dicionario/canais.yaml sem tocar no que foi preenchido à mão.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .binary_format import LdFormatError
from .channel_dictionary import DictionaryFormatError, write_dictionary
from .export import export_session
from .laps import LAP_KIND_TEXT
from .recording import INTERRUPTION_SOURCE_TEXT
from .session import Session


def _format_lap_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}"


def _format_duration(seconds: float) -> str:
    hours, rest = divmod(int(round(seconds)), 3600)
    return f"{hours} h {rest // 60:02d} min {rest % 60:02d} s" if hours else \
        f"{rest // 60} min {rest % 60:02d} s"


def print_summary(session: Session) -> None:
    header = session.header
    print(f"\n=== {session.path.name}")
    print(f"Dispositivo: {header.device_type}  número de série {header.device_serial}  "
          f"versão {header.device_version}")
    print(f"Evento: {header.event}  Sessão: {header.session}  Local: {header.venue}")
    print(f"Piloto: {header.driver}  Veículo: {header.vehicle}")
    identification = ("sem bloco do veículo" if header.vehicle_identification is None
                      else header.vehicle_identification)
    print(f"Equipe: {header.team}  Identificação do veículo: {identification}  Motor: {header.engine}")
    start = session.start_time
    start_text = ("sem horário do GPS" if start is None
                  else f"{start:%d/%m/%Y %H:%M:%S} UTC (GPS)")
    print(f"Início: {start_text}")
    print(f"Duração do log: {session.duration:.1f} s  Canais: {len(session.channels)}")

    missing = sum(interruption.missing_seconds for interruption in session.interruptions)
    print(f"Interrupções de gravação: {len(session.interruptions)}"
          f"  (sem gravação: {_format_duration(missing)})")
    for interruption in session.interruptions:
        position = (f"em {interruption.end:.1f} s" if interruption.located
                    else f"entre {interruption.start:.1f} s e {interruption.end:.1f} s")
        print(f"  {position}  sem gravação por {_format_duration(interruption.missing_seconds)}"
              f"  (fonte: {INTERRUPTION_SOURCE_TEXT[interruption.source]})")

    structure = session.structure
    status = "OK" if structure.data_ends_at_file_end and not structure.gaps else "ATENÇÃO"
    print(f"Estrutura: {status}  (dados terminam no fim do arquivo: "
          f"{structure.data_ends_at_file_end}, lacunas: {len(structure.gaps)})")
    for warning in structure.warnings:
        print(f"  aviso: {warning}")

    print(f"Janelas de perda de dados: {len(session.dropouts)}")
    for window in session.dropouts:
        restart = {True: "reinício da ECU", False: "sem reinício da ECU",
                   None: "reinício da ECU indeterminado"}[window.ecu_restarted]
        speed = ("-" if window.ground_speed_before is None
                 else f"{window.ground_speed_before:.0f} km/h")
        print(f"  {window.start:8.1f} s a {window.end:8.1f} s  "
              f"({window.peak_fraction:.0%} dos canais sem dado, {restart}, "
              f"velocidade antes: {speed})")

    print("Voltas:")
    print(f"  {'número':>6}  {'tipo':<12} {'início (s)':>10} {'oficial':>10} "
          f"{'medida':>10} {'diferença':>9}")
    for lap in session.laps:
        difference = f"{lap.discrepancy:+.3f}" if lap.discrepancy is not None else "-"
        print(f"  {lap.number:>6}  {LAP_KIND_TEXT[lap.kind]:<12} {lap.start:>10.2f} "
              f"{_format_lap_time(lap.official_time):>10} "
              f"{_format_lap_time(lap.measured_duration):>10} {difference:>9}")
    fastest = session.fastest_lap()
    if fastest:
        print(f"Volta mais rápida: volta {fastest.number}, "
              f"{_format_lap_time(fastest.official_time)}")


def _run_dictionary(samples_directory: Path | None, output_path: Path) -> int:
    if samples_directory is None and "MOTEC_AMOSTRAS" in os.environ:
        samples_directory = Path(os.environ["MOTEC_AMOSTRAS"])
    if samples_directory is None or not samples_directory.is_dir():
        print(f"ERRO: pasta de amostras não encontrada ({samples_directory}). "
              "Use --amostras ou defina MOTEC_AMOSTRAS.", file=sys.stderr)
        return 1
    try:
        report = write_dictionary(samples_directory, output_path)
    except (DictionaryFormatError, FileNotFoundError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1
    for rejected in report.rejected:
        print(f"ignorado: {rejected.path.relative_to(samples_directory)} ({rejected.reason})")
    for path in report.accepted:
        print(f"lido:     {path.relative_to(samples_directory)}")
    for name in report.channels_missing_from_logs:
        print(f"aviso: canal '{name}' está no dicionário mas não aparece em nenhum log")
    print(f"Dicionário atualizado: {report.output_path}  "
          f"({len(report.accepted)} sessões, {report.channel_count} canais)")
    return 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="motec_dash")
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("resumo", help="valida e mostra o resumo da sessão")
    summary.add_argument("arquivos", nargs="+", type=Path)
    export = commands.add_parser("exportar", help="exporta para Parquet e JSON")
    export.add_argument("arquivo", type=Path)
    export.add_argument("--saida", type=Path)
    dictionary = commands.add_parser(
        "dicionario", help="gera ou atualiza o rascunho do dicionário de canais")
    dictionary.add_argument("--amostras", type=Path)
    dictionary.add_argument("--saida", type=Path, default=Path("dicionario/canais.yaml"))
    options = parser.parse_args(arguments)

    exit_code = 0
    if options.command == "resumo":
        for path in options.arquivos:
            try:
                with Session(path) as session:
                    print_summary(session)
            except (LdFormatError, KeyError, OSError) as error:
                print(f"\n=== {path.name}\nERRO: {error}", file=sys.stderr)
                exit_code = 1
    elif options.command == "dicionario":
        exit_code = _run_dictionary(options.amostras, options.saida)
    else:
        destination = options.saida or options.arquivo.parent / "processado" / options.arquivo.stem
        with Session(options.arquivo) as session:
            print(f"Exportado para: {export_session(session, destination)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
