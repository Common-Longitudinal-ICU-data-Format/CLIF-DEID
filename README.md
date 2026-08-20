# CLIF-DEID

CLIF-DEID is a configuration-driven tool for de-identifying CLIF 2.1 Parquet
tables before sharing them under an appropriate data use agreement. It uses
Polars for all dataframe and Parquet operations.

CLIF-DEID assists with a defined set of de-identification transformations. It
does not inspect or censor arbitrary free-text fields and does not independently
certify that a dataset is HIPAA compliant or suitable for release. Review the
output under your institution's privacy and governance process.

## Setup

[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

Polars and PyYAML are the runtime dependencies. The project does not provide
pip, Conda, Poetry, or `requirements.txt` workflows.

## Configuration

Start with `config.example.yaml`. Every rule is documented inline. All table
and rule switches must be integer `0` or `1`; YAML booleans are intentionally
rejected. Relative paths are resolved relative to the configuration file.

Every CLIF 2.1 table must have a selection flag. A selected table is read from
`<input_dir>/clif_<table>.parquet` and written with the same filename. The
`patient` and `hospitalization` files are also read as authoritative references
when a selected table requires ID validation, even if those reference tables
are not selected for output.

Version `3.0` is recognized but rejected until its finalized schema is supplied.

## Usage

```bash
uv run clif-deid config.yaml
```

The default shareable output is `CLIF-DEID/De-id`. Existing output is not
replaced unless explicitly requested:

```bash
uv run clif-deid config.yaml --overwrite
```

## Rules

- `replace_patient_id` replaces patient IDs with random 32-character UUID hex strings.
- `replace_hospitalization_id` replaces hospitalization and joined-hospitalization IDs.
- `replace_other_ids` rekeys every other recognized `*_id` field by semantic domain.
- `remove_stray_ids` removes unknown IDs, null IDs, and patient-hospitalization mismatches.
- `remove_rare_diagnoses` removes diagnosis codes found in fewer than the configured number of distinct hospitalizations.
- `cap_age_over_89` replaces ages greater than 89 with 89.
- `null_birth_date` replaces every `birth_date` with null.
- `remove_death_time` truncates non-null `death_dttm` values to midnight.
- `time_shift` assigns one random whole-day offset per patient in the inclusive range `-max_offset_days` through `+max_offset_days`.
- Each geocode switch independently nulls that column while retaining the column and datatype.

Semantic mappings preserve relationships for patient, hospitalization, joined
hospitalization, hospital, procedure, device, medication-order, organism, and
provider IDs. Billing and performing provider IDs share one provider mapping.
Null IDs remain null. An unrecognized `*_id` column stops the run instead of
passing a source identifier through unchanged.

If an ID needed for replacement or time shifting cannot be resolved, the run
fails unless `remove_stray_ids` is enabled to remove that row safely.

## Time Handling

Every Polars `Date` and `Datetime` column associated with a patient is shifted
by that patient's offset. Longitudinal intervals across hospitalizations remain
consistent. When enabled, birth dates are nulled and death datetimes are shifted
before being truncated to midnight.

Parquet timestamps are read exactly as stored. CLIF-DEID performs no timezone
conversion, localization, normalization, replacement, or timezone removal.
Naive timestamps remain naive, and timezone-aware timestamp columns retain
their original timezone metadata. Temporal values stored as strings are not
parsed or modified.

## Private Audit

Each successful run creates `CLIF-DEID/Non-Share/<run_id>/` by default. This
directory contains source identifiers and must never be shared with the
de-identified output. It is excluded by `.gitignore`, and owner-only permissions
are applied where supported.

The private run directory contains:

- `patient_id_mapping.parquet`: original ID, replacement ID, and date offset.
- `hospitalization_id_mapping.parquet`: original/replacement hospitalization and patient IDs.
- One additional mapping file per joined-hospitalization, hospital, procedure, device, medication-order, organism, and provider domain.
- `dropped_rows.parquet`: source table, row number, relevant source identifiers, diagnosis code when applicable, and drop reason.
- `run_summary.json`: per-table flow counts and transformation totals without source identifiers.

Complete dropped clinical rows are not retained. The tool rejects configurations
where the private audit and shareable output directories overlap.

## Tests

Tests create synthetic Parquet data using Polars only:

```bash
uv run python -m unittest discover -s tests -v
```
