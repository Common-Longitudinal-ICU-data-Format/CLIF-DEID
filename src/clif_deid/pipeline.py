from __future__ import annotations

import json
import os
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from clif_deid.config import Config, GEOCODE_COLUMNS


class DeidentificationError(RuntimeError):
    """Raised when de-identification cannot complete safely."""


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    audit_dir: Path


AUDIT_SCHEMA = {
    "table": pl.String,
    "source_row_number": pl.UInt64,
    "patient_id": pl.String,
    "hospitalization_id": pl.String,
    "diagnosis_code_format": pl.String,
    "diagnosis_code": pl.String,
    "drop_reason": pl.String,
}

REQUIRED_IDENTIFIER_COLUMNS = {
    "adt": ("hospitalization_id",),
    "code_status": ("patient_id",),
    "crrt_therapy": ("hospitalization_id",),
    "hospital_diagnosis": ("hospitalization_id",),
    "hospitalization": ("patient_id", "hospitalization_id"),
    "labs": ("hospitalization_id",),
    "medication_admin_continuous": ("hospitalization_id",),
    "medication_admin_intermittent": ("hospitalization_id",),
    "microbiology_culture": ("patient_id", "hospitalization_id", "organism_id"),
    "microbiology_susceptibility": ("organism_id",),
    "patient": ("patient_id",),
    "patient_assessments": ("hospitalization_id",),
    "patient_procedures": ("patient_id", "hospitalization_id"),
    "position": ("hospitalization_id",),
    "respiratory_support": ("hospitalization_id",),
    "vitals": ("hospitalization_id",),
}

ID_COLUMN_DOMAINS = {
    "patient_id": "patient",
    "hospitalization_id": "hospitalization",
    "hospitalization_joined_id": "hospitalization_joined",
    "hospital_id": "hospital",
    "rush_proc_id": "rush_proc",
    "device_id": "device",
    "med_order_id": "med_order",
    "organism_id": "organism",
    "billing_provider_id": "provider",
    "performing_provider_id": "provider",
}

OTHER_ID_DOMAINS = (
    "hospitalization_joined",
    "hospital",
    "rush_proc",
    "device",
    "med_order",
    "organism",
    "provider",
)


def _source_path(config: Config, table: str) -> Path:
    return config.input_dir / f"clif_{table}.parquet"


def _identifier_expr(name: str, dtype: pl.DataType, alias: str) -> pl.Expr:
    value = pl.col(name)
    if dtype.is_float():
        integral_int64 = (
            value.is_finite()
            & (value == value.floor())
            & (value >= -(2**63))
            & (value <= 2**63 - 1)
        )
        normalized = (
            pl.when(integral_int64)
            .then(value.cast(pl.Int64, strict=False).cast(pl.String))
            .otherwise(value.cast(pl.String))
        )
    else:
        normalized = value.cast(pl.String)
    return normalized.alias(alias)


def _schema(path: Path, table: str) -> pl.Schema:
    try:
        return pl.scan_parquet(path).collect_schema()
    except FileNotFoundError as exc:
        raise DeidentificationError(f"Missing input table: {path}") from exc
    except pl.exceptions.PolarsError as exc:
        raise DeidentificationError(f"Cannot read {table} schema from {path}: {exc}") from exc


def _require_columns(
    schema: pl.Schema, columns: tuple[str, ...], table: str
) -> None:
    missing = [column for column in columns if column not in schema]
    if missing:
        raise DeidentificationError(
            f"clif_{table}.parquet is missing required column(s): "
            + ", ".join(missing)
        )


def _collect(lazy: pl.LazyFrame) -> pl.DataFrame:
    return lazy.collect(engine="streaming")


def _row_count(lazy: pl.LazyFrame) -> int:
    return int(_collect(lazy.select(pl.len().alias("count"))).item())


def _condition_count(lazy: pl.LazyFrame, condition: pl.Expr) -> int:
    result = _collect(
        lazy.select(condition.fill_null(False).cast(pl.UInt64).sum().alias("count"))
    ).item()
    return int(result or 0)


def _new_ids(old_ids: list[str], enabled: bool) -> pl.Series:
    if not enabled:
        return pl.Series("new_id", [None] * len(old_ids), dtype=pl.String)
    forbidden = set(old_ids)
    generated: set[str] = set()
    values: list[str] = []
    while len(values) < len(old_ids):
        candidate = uuid.uuid4().hex
        if candidate not in forbidden and candidate not in generated:
            generated.add(candidate)
            values.append(candidate)
    return pl.Series("new_id", values, dtype=pl.String)


def _offsets(count: int, enabled: bool, maximum: int) -> pl.Series:
    values = (
        [secrets.randbelow(2 * maximum + 1) - maximum for _ in range(count)]
        if enabled
        else [None] * count
    )
    return pl.Series("date_shift_days", values, dtype=pl.Int64)


def _build_references(config: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    patient_path = _source_path(config, "patient")
    patient_schema = _schema(patient_path, "patient")
    _require_columns(patient_schema, ("patient_id",), "patient")
    patient = _collect(
        pl.scan_parquet(patient_path).select(
            _identifier_expr(
                "patient_id", patient_schema["patient_id"], "_patient_key"
            )
        )
    )
    if patient["_patient_key"].null_count():
        raise DeidentificationError("clif_patient.parquet contains null patient_id values")
    if patient["_patient_key"].n_unique() != patient.height:
        raise DeidentificationError(
            "clif_patient.parquet contains duplicate patient_id values"
        )

    patient = patient.with_columns(
        pl.col("_patient_key").alias("old_patient_id"),
        _new_ids(
            patient["_patient_key"].to_list(), config.rules.replace_patient_id
        ).alias(
            "new_patient_id"
        ),
        _offsets(
            patient.height, config.rules.time_shift, config.rules.max_offset_days
        ),
        pl.lit(True).alias("_patient_known"),
    )
    if config.rules.replace_patient_id:
        replacements = patient["new_patient_id"]
        if replacements.null_count() or replacements.n_unique() != patient.height:
            raise DeidentificationError("Generated patient ID mappings are not unique")

    hospitalization_path = _source_path(config, "hospitalization")
    hospitalization_schema = _schema(hospitalization_path, "hospitalization")
    _require_columns(
        hospitalization_schema,
        ("hospitalization_id", "patient_id"),
        "hospitalization",
    )
    hospitalization = _collect(
        pl.scan_parquet(hospitalization_path).select(
            _identifier_expr(
                "hospitalization_id",
                hospitalization_schema["hospitalization_id"],
                "_hospitalization_key",
            ),
            _identifier_expr(
                "patient_id", hospitalization_schema["patient_id"], "_patient_key"
            ),
        )
    )
    if hospitalization["_hospitalization_key"].null_count():
        raise DeidentificationError(
            "clif_hospitalization.parquet contains null hospitalization_id values"
        )
    if (
        hospitalization["_hospitalization_key"].n_unique()
        != hospitalization.height
    ):
        raise DeidentificationError(
            "clif_hospitalization.parquet contains duplicate hospitalization_id values"
        )

    hospitalization = hospitalization.with_columns(
        pl.col("_hospitalization_key").alias("old_hospitalization_id"),
        pl.col("_patient_key").alias("old_patient_id"),
        _new_ids(
            hospitalization["_hospitalization_key"].to_list(),
            config.rules.replace_hospitalization_id,
        ).alias("new_hospitalization_id"),
        pl.lit(True).alias("_hospital_known"),
    ).join(
        patient.select(
            "_patient_key",
            "new_patient_id",
            "date_shift_days",
            pl.col("_patient_known").alias("_hospital_patient_valid"),
        ),
        on="_patient_key",
        how="left",
    )
    if config.rules.replace_hospitalization_id:
        replacements = hospitalization["new_hospitalization_id"]
        if replacements.null_count() or replacements.n_unique() != hospitalization.height:
            raise DeidentificationError(
                "Generated hospitalization ID mappings are not unique"
            )
    return patient, hospitalization


def _domain_enabled(domain: str, config: Config) -> bool:
    if domain == "patient":
        return config.rules.replace_patient_id
    if domain in ("hospitalization", "hospitalization_joined"):
        return config.rules.replace_hospitalization_id
    return config.rules.replace_other_ids


def _id_domain(column: str, table: str) -> str:
    try:
        return ID_COLUMN_DOMAINS[column]
    except KeyError as exc:
        raise DeidentificationError(
            f"clif_{table}.parquet contains unrecognized ID column {column}; "
            "add it to a semantic ID domain before processing"
        ) from exc


def _build_other_id_mappings(config: Config) -> dict[str, pl.DataFrame]:
    collected: dict[str, list[pl.DataFrame]] = {
        domain: [] for domain in OTHER_ID_DOMAINS
    }
    for table in config.selected_tables:
        source = _source_path(config, table)
        schema = _schema(source, table)
        for column, dtype in schema.items():
            if not column.endswith("_id"):
                continue
            domain = _id_domain(column, table)
            if domain in ("patient", "hospitalization"):
                continue
            identifiers = _collect(
                pl.scan_parquet(source)
                .select(_identifier_expr(column, dtype, "_id_key"))
                .filter(pl.col("_id_key").is_not_null())
                .unique()
            )
            if identifiers.height:
                collected[domain].append(identifiers)

    mappings: dict[str, pl.DataFrame] = {}
    for domain in OTHER_ID_DOMAINS:
        if collected[domain]:
            identifiers = (
                pl.concat(collected[domain], how="vertical")
                .unique()
                .sort("_id_key")
            )
        else:
            identifiers = pl.DataFrame({"_id_key": pl.Series([], dtype=pl.String)})
        old_ids = identifiers["_id_key"].to_list()
        mappings[domain] = identifiers.with_columns(
            pl.col("_id_key").alias("old_id"),
            _new_ids(old_ids, _domain_enabled(domain, config)).alias("new_id"),
        )
    return mappings


def _attach_references(
    lazy: pl.LazyFrame,
    schema: pl.Schema,
    patient: pl.DataFrame,
    hospitalization: pl.DataFrame,
    other_id_mappings: dict[str, pl.DataFrame],
    config: Config,
    table: str,
) -> tuple[pl.LazyFrame, dict[str, pl.Expr]]:
    expressions: list[pl.Expr] = []
    has_patient = "patient_id" in schema
    has_hospitalization = "hospitalization_id" in schema
    if has_patient:
        expressions.append(
            _identifier_expr(
                "patient_id", schema["patient_id"], "_source_patient_key"
            )
        )
    if has_hospitalization:
        expressions.append(
            _identifier_expr(
                "hospitalization_id",
                schema["hospitalization_id"],
                "_source_hospitalization_key",
            )
        )
    if expressions:
        lazy = lazy.with_columns(expressions)

    if has_patient:
        patient_lookup = patient.lazy().select(
            pl.col("_patient_key").alias("_lookup_patient_key"),
            pl.col("_patient_known").alias("_direct_patient_known"),
            pl.col("new_patient_id").alias("_new_patient_id"),
            pl.col("date_shift_days").alias("_direct_date_shift_days"),
        )
        lazy = lazy.join(
            patient_lookup,
            left_on="_source_patient_key",
            right_on="_lookup_patient_key",
            how="left",
        )

    for column, dtype in schema.items():
        if not column.endswith("_id") or column in (
            "patient_id",
            "hospitalization_id",
        ):
            continue
        domain = _id_domain(column, table)
        if not _domain_enabled(domain, config):
            continue
        source_key = f"_source_rekey_{column}"
        new_key = f"_new_rekey_{column}"
        lookup_key = f"_lookup_rekey_{column}"
        lazy = lazy.with_columns(_identifier_expr(column, dtype, source_key))
        lookup = other_id_mappings[domain].lazy().select(
            pl.col("_id_key").alias(lookup_key),
            pl.col("new_id").alias(new_key),
        )
        lazy = lazy.join(
            lookup,
            left_on=source_key,
            right_on=lookup_key,
            how="left",
        )

    if has_hospitalization:
        hospital_lookup = hospitalization.lazy().select(
            pl.col("_hospitalization_key").alias("_lookup_hospitalization_key"),
            pl.col("_hospital_known"),
            pl.col("_hospital_patient_valid"),
            pl.col("_patient_key").alias("_hospital_patient_key"),
            pl.col("new_hospitalization_id").alias("_new_hospitalization_id"),
            pl.col("date_shift_days").alias("_hospital_date_shift_days"),
        )
        lazy = lazy.join(
            hospital_lookup,
            left_on="_source_hospitalization_key",
            right_on="_lookup_hospitalization_key",
            how="left",
        )

    patient_missing = (
        pl.col("_source_patient_key").is_null() if has_patient else pl.lit(False)
    )
    patient_unknown = (
        pl.col("_source_patient_key").is_not_null()
        & pl.col("_direct_patient_known").is_null()
        if has_patient
        else pl.lit(False)
    )
    hospitalization_missing = (
        pl.col("_source_hospitalization_key").is_null()
        if has_hospitalization
        else pl.lit(False)
    )
    hospitalization_unknown = (
        pl.col("_source_hospitalization_key").is_not_null()
        & pl.col("_hospital_known").is_null()
        if has_hospitalization
        else pl.lit(False)
    )
    hospitalization_patient_unknown = (
        pl.col("_hospital_known").is_not_null()
        & pl.col("_hospital_patient_valid").is_null()
        if has_hospitalization
        else pl.lit(False)
    )
    mismatch = (
        pl.col("_direct_patient_known").is_not_null()
        & pl.col("_hospital_patient_valid").is_not_null()
        & (pl.col("_source_patient_key") != pl.col("_hospital_patient_key"))
        if has_patient and has_hospitalization
        else pl.lit(False)
    )
    stray = (
        patient_missing
        | patient_unknown
        | hospitalization_missing
        | hospitalization_unknown
        | hospitalization_patient_unknown
        | mismatch
    )

    reasons = pl.concat_str(
        [
            pl.when(patient_missing)
            .then(pl.lit("missing_patient_id"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(patient_unknown)
            .then(pl.lit("unknown_patient_id"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(hospitalization_missing)
            .then(pl.lit("missing_hospitalization_id"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(hospitalization_unknown)
            .then(pl.lit("unknown_hospitalization_id"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(hospitalization_patient_unknown)
            .then(pl.lit("hospitalization_has_unknown_patient"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(mismatch)
            .then(pl.lit("patient_hospitalization_mismatch"))
            .otherwise(pl.lit(None, dtype=pl.String)),
        ],
        separator="|",
        ignore_nulls=True,
    )

    if has_patient:
        resolved_patient = pl.col("_source_patient_key")
        resolved_offset = pl.col("_direct_date_shift_days")
    elif has_hospitalization:
        resolved_patient = pl.col("_hospital_patient_key")
        resolved_offset = pl.col("_hospital_date_shift_days")
    else:
        resolved_patient = pl.lit(None, dtype=pl.String)
        resolved_offset = pl.lit(None, dtype=pl.Int64)
    lazy = lazy.with_columns(
        resolved_patient.alias("_resolved_patient_key"),
        resolved_offset.alias("_date_shift_days"),
    )
    return lazy, {
        "stray": stray,
        "reasons": reasons,
        "has_patient": pl.lit(has_patient),
        "has_hospitalization": pl.lit(has_hospitalization),
    }


def _audit_frame(
    lazy: pl.LazyFrame,
    table: str,
    schema: pl.Schema,
    reason: pl.Expr,
) -> pl.LazyFrame:
    return lazy.select(
        pl.lit(table).alias("table"),
        pl.col("_source_row_number").cast(pl.UInt64).alias("source_row_number"),
        (
            pl.col("_source_patient_key")
            if "patient_id" in schema
            else pl.lit(None, dtype=pl.String)
        ).alias("patient_id"),
        (
            pl.col("_source_hospitalization_key")
            if "hospitalization_id" in schema
            else pl.lit(None, dtype=pl.String)
        ).alias("hospitalization_id"),
        (
            pl.col("diagnosis_code_format").cast(pl.String)
            if "diagnosis_code_format" in schema
            else pl.lit(None, dtype=pl.String)
        ).alias("diagnosis_code_format"),
        (
            pl.col("diagnosis_code").cast(pl.String)
            if "diagnosis_code" in schema
            else pl.lit(None, dtype=pl.String)
        ).alias("diagnosis_code"),
        reason.alias("drop_reason"),
    )


def _temporal_columns(schema: pl.Schema) -> list[str]:
    return [
        name
        for name, dtype in schema.items()
        if dtype == pl.Date or isinstance(dtype, pl.Datetime)
    ]


def _output_frame(
    lazy: pl.LazyFrame, schema: pl.Schema, config: Config
) -> pl.LazyFrame:
    output: list[pl.Expr] = []
    temporal = set(_temporal_columns(schema))
    for name, dtype in schema.items():
        if name == "patient_id" and config.rules.replace_patient_id:
            expression = pl.col("_new_patient_id").cast(pl.String)
        elif name == "hospitalization_id" and config.rules.replace_hospitalization_id:
            expression = pl.col("_new_hospitalization_id").cast(pl.String)
        elif name.endswith("_id") and _domain_enabled(
            ID_COLUMN_DOMAINS[name], config
        ):
            expression = pl.col(f"_new_rekey_{name}").cast(pl.String)
        elif name == "birth_date" and config.rules.null_birth_date:
            expression = pl.lit(None, dtype=dtype)
        elif (
            name == "age_at_admission"
            and config.rules.cap_age_over_89
            and dtype.is_numeric()
        ):
            expression = pl.when(pl.col(name) > 89).then(pl.lit(89)).otherwise(
                pl.col(name)
            )
        elif name in GEOCODE_COLUMNS and config.rules.geocode[name]:
            expression = pl.lit(None, dtype=dtype)
        elif name in temporal and (
            config.rules.time_shift
            or (name == "death_dttm" and config.rules.remove_death_time)
        ):
            expression = pl.col(name)
            if config.rules.time_shift:
                time_unit = dtype.time_unit if isinstance(dtype, pl.Datetime) else None
                expression = expression + pl.duration(
                    days=pl.col("_date_shift_days"), time_unit=time_unit
                )
            if (
                name == "death_dttm"
                and config.rules.remove_death_time
                and isinstance(dtype, pl.Datetime)
            ):
                expression = expression.dt.truncate("1d")
        else:
            expression = pl.col(name)
        output.append(expression.alias(name))
    return lazy.select(output)


def _metric_counts(
    lazy: pl.LazyFrame, schema: pl.Schema, config: Config
) -> dict[str, int]:
    expressions: list[pl.Expr] = [pl.len().alias("output_rows")]
    for column in schema:
        if not column.endswith("_id"):
            continue
        domain = ID_COLUMN_DOMAINS[column]
        if _domain_enabled(domain, config):
            expressions.append(
                pl.col(column).is_not_null().sum().alias(f"{column}_rekeyed")
            )
    if "age_at_admission" in schema and config.rules.cap_age_over_89:
        expressions.append(
            (pl.col("age_at_admission") > 89).sum().alias("ages_capped")
        )
    if "birth_date" in schema and config.rules.null_birth_date:
        expressions.append(
            pl.col("birth_date").is_not_null().sum().alias("birth_dates_nulled")
        )
    if "death_dttm" in schema and config.rules.remove_death_time:
        expressions.append(
            pl.col("death_dttm")
            .is_not_null()
            .sum()
            .alias("death_datetimes_truncated")
        )
    for column in GEOCODE_COLUMNS:
        if column in schema and config.rules.geocode[column]:
            expressions.append(
                pl.col(column).is_not_null().sum().alias(f"{column}_nulled")
            )
    temporal = [
        column
        for column in _temporal_columns(schema)
        if not (column == "birth_date" and config.rules.null_birth_date)
    ]
    if temporal and config.rules.time_shift:
        shifted = [
            (
                pl.col(column).is_not_null()
                & pl.col("_date_shift_days").is_not_null()
            )
            .cast(pl.UInt64)
            .sum()
            for column in temporal
        ]
        expressions.append(pl.sum_horizontal(shifted).alias("timestamps_shifted"))
    values = _collect(lazy.select(expressions)).row(0, named=True)
    return {name: int(value or 0) for name, value in values.items()}


def _validate_paths(config: Config, overwrite: bool) -> None:
    if not config.input_dir.is_dir():
        raise DeidentificationError(f"input_dir is not a directory: {config.input_dir}")
    if config.input_dir == config.output_dir:
        raise DeidentificationError("input_dir and output_dir cannot be the same")
    output = config.output_dir
    private = config.non_share_dir
    if output in config.input_dir.parents:
        raise DeidentificationError("output_dir cannot contain input_dir")
    if private == config.input_dir or private in config.input_dir.parents:
        raise DeidentificationError("non_share_dir cannot contain input_dir")
    if output == private or output in private.parents or private in output.parents:
        raise DeidentificationError(
            "output_dir and non_share_dir cannot overlap or contain one another"
        )
    if output.exists() and not overwrite:
        raise DeidentificationError(
            f"output_dir already exists: {output}; use --overwrite to replace it"
        )
    if output.exists() and not output.is_dir():
        raise DeidentificationError(f"output_dir exists but is not a directory: {output}")
    for table in config.selected_tables:
        path = _source_path(config, table)
        if not path.is_file():
            raise DeidentificationError(f"Selected input table is missing: {path}")


def _make_staging_dir(path: Path, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=mode)


def _promote_directory(stage: Path, target: Path, overwrite: bool) -> None:
    if not target.exists():
        os.replace(stage, target)
        return
    if not overwrite:
        raise DeidentificationError(f"Refusing to overwrite existing directory: {target}")
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.backup")
    os.replace(target, backup)
    try:
        os.replace(stage, target)
    except Exception:
        os.replace(backup, target)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        raise DeidentificationError(
            f"New output was installed, but the previous output backup could not "
            f"be removed: {backup}: {exc}"
        ) from exc


def _restrict_private_permissions(root: Path) -> None:
    try:
        os.chmod(root, 0o700)
        for path in root.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    except OSError as exc:
        raise DeidentificationError(
            f"Could not restrict private audit permissions in {root}: {exc}"
        ) from exc


def _write_private_mappings(
    patient: pl.DataFrame,
    hospitalization: pl.DataFrame,
    other_id_mappings: dict[str, pl.DataFrame],
    audit_stage: Path,
) -> None:
    patient.select(
        "old_patient_id", "new_patient_id", "date_shift_days"
    ).write_parquet(audit_stage / "patient_id_mapping.parquet")
    hospitalization.select(
        "old_hospitalization_id",
        "new_hospitalization_id",
        "old_patient_id",
        "new_patient_id",
    ).write_parquet(audit_stage / "hospitalization_id_mapping.parquet")
    for domain in OTHER_ID_DOMAINS:
        other_id_mappings[domain].select("old_id", "new_id").write_parquet(
            audit_stage / f"{domain}_id_mapping.parquet"
        )


def run(config: Config, overwrite: bool = False) -> RunResult:
    _validate_paths(config, overwrite)
    patient, hospitalization = _build_references(config)
    other_id_mappings = _build_other_id_mappings(config)

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output_stage = config.output_dir.parent / f".{config.output_dir.name}.{run_id}.tmp"
    audit_stage = config.non_share_dir / f".{run_id}.tmp"
    audit_target = config.non_share_dir / run_id
    if output_stage.exists() or audit_stage.exists() or audit_target.exists():
        raise DeidentificationError("Generated staging path already exists; run again")

    _make_staging_dir(output_stage)
    _make_staging_dir(audit_stage, mode=0o700)
    drop_frames: list[pl.LazyFrame] = []
    table_summary: dict[str, dict[str, Any]] = {}

    try:
        _write_private_mappings(
            patient, hospitalization, other_id_mappings, audit_stage
        )
        for table in config.selected_tables:
            source = _source_path(config, table)
            schema = _schema(source, table)
            _require_columns(schema, REQUIRED_IDENTIFIER_COLUMNS[table], table)
            lazy = pl.scan_parquet(source).with_row_index("_source_row_number")
            enriched, conditions = _attach_references(
                lazy,
                schema,
                patient,
                hospitalization,
                other_id_mappings,
                config,
                table,
            )
            if (
                table == "patient"
                and config.rules.remove_death_time
                and "death_dttm" in schema
                and not isinstance(schema["death_dttm"], pl.Datetime)
            ):
                raise DeidentificationError(
                    "clif_patient.parquet death_dttm must be a Datetime when "
                    "remove_death_time is enabled"
                )
            for column in schema:
                if not column.endswith("_id") or column in (
                    "patient_id",
                    "hospitalization_id",
                ):
                    continue
                domain = _id_domain(column, table)
                if not _domain_enabled(domain, config):
                    continue
                unresolved_mapping = (
                    pl.col(f"_source_rekey_{column}").is_not_null()
                    & pl.col(f"_new_rekey_{column}").is_null()
                )
                unresolved_mapping_count = _condition_count(
                    enriched, unresolved_mapping
                )
                if unresolved_mapping_count:
                    raise DeidentificationError(
                        f"clif_{table}.parquet has {unresolved_mapping_count} "
                        f"{column} value(s) without a private rekey mapping"
                    )
            input_rows = _row_count(enriched)
            stray_count = _condition_count(enriched, conditions["stray"])

            relationship_rules = (
                config.rules.replace_patient_id
                or config.rules.replace_hospitalization_id
                or config.rules.time_shift
            )
            if stray_count and not config.rules.remove_stray_ids and relationship_rules:
                raise DeidentificationError(
                    f"clif_{table}.parquet has {stray_count} unresolved or mismatched "
                    "identifier row(s); enable remove_stray_ids to process safely"
                )

            if config.rules.remove_stray_ids and stray_count:
                drop_frames.append(
                    _audit_frame(
                        enriched.filter(conditions["stray"]),
                        table,
                        schema,
                        conditions["reasons"],
                    )
                )
                current = enriched.filter(~conditions["stray"].fill_null(False))
            else:
                current = enriched
                stray_count = 0

            temporal = _temporal_columns(schema)
            if config.rules.time_shift and temporal:
                unresolved_temporal = pl.any_horizontal(
                    [pl.col(column).is_not_null() for column in temporal]
                ) & pl.col("_date_shift_days").is_null()
                unresolved_count = _condition_count(current, unresolved_temporal)
                if unresolved_count:
                    raise DeidentificationError(
                        f"clif_{table}.parquet has {unresolved_count} row(s) with dates "
                        "that cannot be associated with a patient"
                    )

            rare_count = 0
            if table == "hospital_diagnosis" and config.rules.remove_rare_diagnoses:
                _require_columns(
                    schema,
                    (
                        "hospitalization_id",
                        "diagnosis_code_format",
                        "diagnosis_code",
                    ),
                    table,
                )
                frequencies = (
                    current.filter(
                        pl.col("diagnosis_code_format").is_not_null()
                        & pl.col("diagnosis_code").is_not_null()
                    )
                    .group_by("diagnosis_code_format", "diagnosis_code")
                    .agg(
                        pl.col("_source_hospitalization_key")
                        .n_unique()
                        .alias("_diagnosis_hospitalizations")
                    )
                )
                current = current.join(
                    frequencies,
                    on=["diagnosis_code_format", "diagnosis_code"],
                    how="left",
                )
                rare = (
                    pl.col("_diagnosis_hospitalizations").is_not_null()
                    & (
                        pl.col("_diagnosis_hospitalizations")
                        < config.rules.diagnosis_min_hospitalizations
                    )
                )
                rare_count = _condition_count(current, rare)
                if rare_count:
                    drop_frames.append(
                        _audit_frame(
                            current.filter(rare),
                            table,
                            schema,
                            pl.lit("rare_diagnosis_code"),
                        )
                    )
                    current = current.filter(~rare.fill_null(False))

            metrics = _metric_counts(current, schema, config)
            output = _output_frame(current, schema, config)
            output_path = output_stage / f"clif_{table}.parquet"
            output.sink_parquet(output_path, mkdir=True)

            missing_geocodes = [
                column
                for column in GEOCODE_COLUMNS
                if config.rules.geocode[column] and column not in schema
            ]
            table_summary[table] = {
                "input_rows": input_rows,
                "stray_rows_dropped": stray_count,
                "rare_diagnosis_rows_dropped": rare_count,
                **metrics,
                "missing_requested_geocode_columns": missing_geocodes,
            }

        dropped_path = audit_stage / "dropped_rows.parquet"
        drop_reasons: dict[str, int] = {}
        if drop_frames:
            dropped = pl.concat(drop_frames, how="vertical")
            reason_counts = _collect(
                dropped.select(
                    "table",
                    pl.col("drop_reason").str.split("|").alias("drop_reason"),
                )
                .explode("drop_reason", empty_as_null=True)
                .group_by("table", "drop_reason")
                .len()
            )
            for row in reason_counts.iter_rows(named=True):
                table = str(row["table"])
                reason = str(row["drop_reason"])
                count = int(row["len"])
                table_summary[table].setdefault("drop_reasons", {})[reason] = count
                drop_reasons[reason] = drop_reasons.get(reason, 0) + count
            dropped.sink_parquet(dropped_path, mkdir=True)
        else:
            pl.DataFrame(schema=AUDIT_SCHEMA).write_parquet(dropped_path)
        for summary in table_summary.values():
            summary.setdefault("drop_reasons", {})

        totals: dict[str, int] = {}
        for summary in table_summary.values():
            for key, value in summary.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        report = {
            "run_id": run_id,
            "clif_version": config.version,
            "selected_tables": list(config.selected_tables),
            "tables": table_summary,
            "totals": totals,
            "drop_reasons": drop_reasons,
            "id_domain_mappings": {
                "patient": patient.height,
                "hospitalization": hospitalization.height,
                **{
                    domain: mapping.height
                    for domain, mapping in other_id_mappings.items()
                },
            },
            "output_dir": str(config.output_dir),
            "non_share_run_dir": str(audit_target),
        }
        with (audit_stage / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

        _restrict_private_permissions(audit_stage)
        config.non_share_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(config.non_share_dir, 0o700)
        os.replace(audit_stage, audit_target)
        try:
            _promote_directory(output_stage, config.output_dir, overwrite)
        except Exception:
            if output_stage.exists():
                shutil.rmtree(audit_target)
            raise
    except Exception:
        if output_stage.exists():
            shutil.rmtree(output_stage)
        if audit_stage.exists():
            shutil.rmtree(audit_stage)
        raise

    return RunResult(output_dir=config.output_dir, audit_dir=audit_target)
