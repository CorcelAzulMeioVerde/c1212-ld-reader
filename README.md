**English** | [Português](README.pt-BR.md)

# motec_dash — reader for MoTeC C1212 dash `.ld` logs

Reads the `.ld` file recorded by the C1212 dash logger, validates its structure, flags
missing data, segments laps, rebuilds UTC time from the GPS clock and exports to Parquet.
It does not use the `.ldx` file.

> **Independent, unofficial project.** It is not affiliated with, endorsed, maintained
> or supported by MoTeC. "MoTeC" and "C1212" are mentioned only to state which hardware
> the reader is compatible with. The `.ld` format has no public documentation: the layout
> used here was obtained by reverse engineering, for interoperability, and may not hold
> for other devices or firmware versions.

The command-line interface, the channel dictionary keys and the code comments are in
Portuguese; the Python API and the exported column and metadata names are in English.

## Installation and usage

    pip install .
    python -m motec_dash resumo   file.ld                   # summary
    python -m motec_dash exportar file.ld [--saida FOLDER]  # export

Without `--saida`, the export goes to `<.ld folder>/processado/<file name>/`:
one Parquet file per sampling frequency (`100Hz.parquet`, `20Hz.parquet`, ...),
`lap_statistics.parquet` (minimum, maximum, mean and valid sample count per lap and channel)
and `metadata.json` (header, SHA-256 of the original file, laps, data dropouts, channels).

In Python:

    from motec_dash import Session
    with Session("file.ld") as session:
        throttle = session.channel("Throttle Position")   # .time, .values (NaN = missing)
        laps = session.timed_laps()

Channel names come from the dash configuration; the ones used by the rules below
(`Running Lap Time`, `Lap Time`, `Lap Number`, `GPS Time`, `GPS Date`, `GPS Quality`,
`GPS Latitude`, `GPS Longitude`, `ECU Uptime`, `Ground Speed`) must exist with these names.

## Missing-data rules

1. A raw value of -1 is missing only inside windows where most channels read -1 at the
   same time (communication loss). Outside them, -1 is a legitimate value (e.g. -0.01 G).
   The majority is measured across all channels and also within each sampling frequency:
   very short dropouts only show up in the fast channels.
2. GPS position and speed channels become missing when `GPS Quality` = 0 or latitude and
   longitude are both zero.

## Recording interruptions and absolute time

The dash can stop recording without marking it in the file: the time computed from the
sample index ("log time") is not real time.

- The time reference is the GPS clock: `GPS Date` (ddmmyy) and `GPS Time`
  (hhmmss.s, 0.1 s resolution), in UTC. It is only valid with a position fix. `GPS Date`
  may be 1024 weeks behind (GPS week-number rollover); dates before 2020 are corrected.
- `session.interruptions`: for each interruption, its position in the log (`start`, `end`),
  real duration (`missing_seconds`) and source (`gps` or `lap_timer`). The jump in
  `Running Lap Time` is only a fallback where the GPS has no fix: with the timer stopped
  at zero it shows nothing.
- **Unlocated** interruption (`located` false): it fell in a stretch without a GPS fix.
  How much time is missing is known, but not at which sample; samples in that stretch
  have no timestamp.
- `session.absolute_time(times)`: UTC time (`datetime64[ms]`) for each log time, with one
  anchor per continuous recording segment.
- `session.start_time`: first valid GPS time. The header date and time come from the dash
  clock, which is unreliable, and are not used in any calculation (the export keeps them
  only as `dash_clock`).
- The Parquet files have a `gps_time` (UTC) column next to `log_time` (log time).

## Laps

- `Lap Number` defines how many crossings happened; `Running Lap Time` (100 Hz) defines
  when; `Lap Time` (the dash's official lap time) compensates for the timer delay.
- The first crossing (timer stopped at zero) is derived: start = end - official time.
- Kinds (`Lap.kind`): `out` (out lap: from the start of the log to the first crossing),
  `timed`, `interrupted` and `incomplete`. The Portuguese text appears only in the output
  of the summary command (`LAP_KIND_TEXT`).
- The dash may stop recording and come back with `Lap Number` reset to zero. The lap number
  is the order of the crossings; the dash's own number is kept in `dash_number`. A lap that
  contains a recording interruption is `interrupted` and is not returned by `timed_laps()`.
- Data dropout followed by a recording interruption: whether the ECU restarted is
  undetermined (`ecu_restarted` = `None`); the speed before the dropout is kept.

## Channel dictionary draft

    python -m motec_dash dicionario --amostras FOLDER [--saida dicionario/canais.yaml]

Reads every C1212 dash log in the folder (and subfolders; logs from other devices and
copies with the same SHA-256 are skipped) and writes, for each channel, the `observado`
(observed) block: original name, short name, unit, frequency, data type, per-session and
overall statistics (missing samples already excluded), time spent at each value for
channels with up to 20 distinct values, and whether the channel is constant. Without
`--amostras`, it uses the `MOTEC_AMOSTRAS` environment variable.

If the file already exists, only `observado` (in each channel) and `sessoes` (sessions)
are updated. Everything else is filled in by people and is never overwritten:
`nome_canonico` (canonical name), `descricao` (description), `significado_dos_estados`
(meaning of states), `limites` (limits), `prioridade` (priority), extra keys and comments.
A file that is not valid YAML stops the run without writing anything.

## Tests

    pip install ".[testes]"
    python -m pytest

The tests use no real logs: `tests/ld_sintetico.py` generates a `.ld` file with a made-up
session (data dropout, GPS without a fix, recording interruption and known laps).

## Development

This project was developed with [Claude Code](https://claude.com/claude-code), Anthropic's
AI coding tool, under the author's direction and review.

## License

MIT. Copyright (c) 2026 Heitor Rodrigues de Farias. See [LICENSE](LICENSE).
