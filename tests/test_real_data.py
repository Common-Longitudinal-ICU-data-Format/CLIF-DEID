from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl

from clif_deid.config import GEOCODE_COLUMNS, TABLES, load_config
from clif_deid.pipeline import ID_COLUMN_DOMAINS, run


REAL_DATA_ENV = "CLIF_DEID_REAL_DATA_DIR"
TEMP_DIR_ENV = "CLIF_DEID_REAL_TEST_TMPDIR"
DIAGNOSIS_MIN_HOSPITALIZATIONS = 10
MAX_OFFSET_DAYS = 45


def _count(lazy: pl.LazyFrame) -> int:
    return int(lazy.select(pl.len()).collect(engine="streaming").item())


def _normalized_identifier(name: str, dtype: pl.DataType, alias: str) -> pl.Expr:
    value = pl.col(name)
    if dtype.is_float():
        integral = (
            value.is_finite()
            & (value == value.floor())
            & (value >= -(2**63))
            & (value <= 2**63 - 1)
        )
        value = (
            pl.when(integral)
            .then(value.cast(pl.Int64, strict=False).cast(pl.String))
            .otherwise(value.cast(pl.String))
        )
    else:
        value = value.cast(pl.String)
    return value.alias(alias)


def _write_config(root: Path, input_dir: Path) -> Path:
    tables = "\n".join(f"  {table}: 1" for table in TABLES)
    geocodes = "\n".join(f"    {column}: 1" for column in GEOCODE_COLUMNS)
    text = f'''version: "2.1"
input_dir: "{input_dir}"
output_dir: "{root / 'output'}"
non_share_dir: "{root / 'private'}"

tables:
{tables}

rules:
  replace_patient_id: 1
  replace_hospitalization_id: 1
  replace_other_ids: 1
  remove_stray_ids: 1
  remove_rare_diagnoses: 1
  diagnosis_min_hospitalizations: {DIAGNOSIS_MIN_HOSPITALIZATIONS}
  cap_age_over_89: 1
  null_birth_date: 1
  remove_death_time: 1
  time_shift: 1
  max_offset_days: {MAX_OFFSET_DAYS}
  geocode:
{geocodes}
'''
    path = root / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@unittest.skipUnless(
    os.environ.get(REAL_DATA_ENV),
    f"set {REAL_DATA_ENV} to run private real-data integration tests",
)
class RealDataIntegrityTests(unittest.TestCase):
    """End-to-end integrity checks that never report row-level private data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_dir = Path(os.environ[REAL_DATA_ENV]).expanduser().resolve()
        missing = [
            table
            for table in TABLES
            if not (cls.input_dir / f"clif_{table}.parquet").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{REAL_DATA_ENV} is missing {len(missing)} supported table(s): "
                + ", ".join(missing)
            )

        temporary_parent = os.environ.get(TEMP_DIR_ENV)
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="clif-deid-real-", dir=temporary_parent
        )
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.root = Path(cls._temporary.name)
        cls.source_stats = {
            table: (cls._source_path(table).stat().st_size, cls._source_path(table).stat().st_mtime_ns)
            for table in TABLES
        }
        cls.result = run(load_config(_write_config(cls.root, cls.input_dir)))
        cls.summary = json.loads(
            (cls.result.audit_dir / "run_summary.json").read_text(encoding="utf-8")
        )

    @classmethod
    def _source_path(cls, table: str) -> Path:
        return cls.input_dir / f"clif_{table}.parquet"

    @classmethod
    def _output_path(cls, table: str) -> Path:
        return cls.result.output_dir / f"clif_{table}.parquet"

    @classmethod
    def _mapping(cls, column: str) -> pl.LazyFrame:
        domain = ID_COLUMN_DOMAINS[column]
        if domain == "patient":
            return pl.scan_parquet(
                cls.result.audit_dir / "patient_id_mapping.parquet"
            ).select(
                pl.col("old_patient_id").alias("_old"),
                pl.col("new_patient_id").alias("_new"),
            )
        if domain == "hospitalization":
            return pl.scan_parquet(
                cls.result.audit_dir / "hospitalization_id_mapping.parquet"
            ).select(
                pl.col("old_hospitalization_id").alias("_old"),
                pl.col("new_hospitalization_id").alias("_new"),
            )
        return pl.scan_parquet(
            cls.result.audit_dir / f"{domain}_id_mapping.parquet"
        ).select(pl.col("old_id").alias("_old"), pl.col("new_id").alias("_new"))

    @classmethod
    def _retained_source(cls, table: str) -> pl.LazyFrame:
        dropped = (
            pl.scan_parquet(cls.result.audit_dir / "dropped_rows.parquet")
            .filter(pl.col("table") == table)
            .select("source_row_number")
            .unique()
        )
        return (
            pl.scan_parquet(cls._source_path(table))
            .with_row_index("source_row_number")
            .join(dropped, on="source_row_number", how="anti")
        )

    @classmethod
    def _expected_identifiers(
        cls, table: str
    ) -> tuple[pl.LazyFrame, list[str], pl.Schema]:
        schema = pl.scan_parquet(cls._source_path(table)).collect_schema()
        identifiers = [name for name in schema if name.endswith("_id")]
        lazy = cls._retained_source(table)
        for index, column in enumerate(identifiers):
            old = f"_old_id_{index}"
            expected = f"_expected_id_{index}"
            lazy = lazy.with_columns(
                _normalized_identifier(column, schema[column], old)
            ).join(
                cls._mapping(column).rename(
                    {"_old": f"_lookup_id_{index}", "_new": expected}
                ),
                left_on=old,
                right_on=f"_lookup_id_{index}",
                how="left",
            )
        return lazy, identifiers, schema

    def assert_zero(self, value: int, message: str) -> None:
        self.assertEqual(value, 0, f"{message}: {value} violating row(s)")

    def test_row_and_audit_accounting(self) -> None:
        self.assertEqual(self.summary["selected_tables"], list(TABLES))
        dropped = pl.scan_parquet(self.result.audit_dir / "dropped_rows.parquet")
        duplicate_audit_rows = _count(
            dropped.group_by("table", "source_row_number").len().filter(pl.col("len") > 1)
        )
        self.assert_zero(duplicate_audit_rows, "duplicate dropped-row audit entries")

        reason_rows = (
            dropped.select(
                "table", pl.col("drop_reason").str.split("|").alias("drop_reason")
            )
            .explode("drop_reason", empty_as_null=True)
            .group_by("table", "drop_reason")
            .len()
            .collect(engine="streaming")
        )
        actual_global_reasons: dict[str, int] = {}
        actual_table_reasons: dict[str, dict[str, int]] = {}
        for row in reason_rows.iter_rows(named=True):
            table = str(row["table"])
            reason = str(row["drop_reason"])
            count = int(row["len"])
            actual_table_reasons.setdefault(table, {})[reason] = count
            actual_global_reasons[reason] = actual_global_reasons.get(reason, 0) + count
        self.assertEqual(actual_global_reasons, self.summary["drop_reasons"])

        for table in TABLES:
            source_rows = _count(pl.scan_parquet(self._source_path(table)))
            output_rows = _count(pl.scan_parquet(self._output_path(table)))
            audit_rows = _count(dropped.filter(pl.col("table") == table))
            reported = self.summary["tables"][table]
            self.assertEqual(reported["input_rows"], source_rows, table)
            self.assertEqual(reported["output_rows"], output_rows, table)
            self.assertEqual(source_rows, output_rows + audit_rows, table)
            self.assertEqual(
                audit_rows,
                reported["stray_rows_dropped"]
                + reported["rare_diagnosis_rows_dropped"],
                table,
            )
            self.assertEqual(
                actual_table_reasons.get(table, {}), reported["drop_reasons"], table
            )
            invalid_audit_indexes = _count(
                dropped.filter(
                    (pl.col("table") == table)
                    & (pl.col("source_row_number") >= source_rows)
                )
            )
            self.assert_zero(invalid_audit_indexes, f"{table} invalid audit indexes")

    def test_mapping_bijections_and_referential_closure(self) -> None:
        mapping_specs = {
            "patient": ("old_patient_id", "new_patient_id"),
            "hospitalization": (
                "old_hospitalization_id",
                "new_hospitalization_id",
            ),
            **{
                domain: ("old_id", "new_id")
                for domain in set(ID_COLUMN_DOMAINS.values())
                - {"patient", "hospitalization"}
            },
        }
        for domain, (old, new) in mapping_specs.items():
            mapping_path = self.result.audit_dir / f"{domain}_id_mapping.parquet"
            mapping = pl.scan_parquet(mapping_path)
            rows = _count(mapping)
            self.assertEqual(rows, self.summary["id_domain_mappings"][domain], domain)
            self.assert_zero(
                _count(mapping.filter(pl.col(old).is_null() | pl.col(new).is_null())),
                f"{domain} null mappings",
            )
            self.assertEqual(
                rows,
                int(mapping.select(pl.col(old).n_unique()).collect(engine="streaming").item()),
                f"{domain} old mapping uniqueness",
            )
            self.assertEqual(
                rows,
                int(mapping.select(pl.col(new).n_unique()).collect(engine="streaming").item()),
                f"{domain} new mapping uniqueness",
            )
            self.assert_zero(
                _count(
                    mapping.select(pl.col(old).alias("_value")).join(
                        mapping.select(pl.col(new).alias("_value")),
                        on="_value",
                        how="inner",
                    )
                ),
                f"{domain} source identifier leakage",
            )
            invalid_replacements = _count(
                mapping.filter(~pl.col(new).str.contains(r"^[0-9a-f]{32}$"))
            )
            self.assert_zero(invalid_replacements, f"{domain} invalid replacements")

        patient_map = pl.scan_parquet(
            self.result.audit_dir / "patient_id_mapping.parquet"
        )
        offsets_out_of_bounds = _count(
            patient_map.filter(
                pl.col("date_shift_days").is_null()
                | (pl.col("date_shift_days").abs() > MAX_OFFSET_DAYS)
            )
        )
        self.assert_zero(offsets_out_of_bounds, "patient date offsets out of bounds")

        hospital_map = pl.scan_parquet(
            self.result.audit_dir / "hospitalization_id_mapping.parquet"
        )
        wrong_hospital_owners = _count(
            hospital_map.join(
                patient_map.select(
                    "old_patient_id",
                    pl.col("new_patient_id").alias("_expected_patient_id"),
                ),
                on="old_patient_id",
                how="left",
            ).filter(
                ~pl.col("new_patient_id").eq_missing(pl.col("_expected_patient_id"))
            )
        )
        self.assert_zero(wrong_hospital_owners, "hospitalization mapping ownership")

        patients = pl.scan_parquet(self._output_path("patient")).select("patient_id")
        hospitals = pl.scan_parquet(self._output_path("hospitalization")).select(
            "hospitalization_id", pl.col("patient_id").alias("_hospital_patient_id")
        )
        self.assert_zero(
            _count(
                hospitals.join(
                    patients, left_on="_hospital_patient_id", right_on="patient_id", how="anti"
                )
            ),
            "hospitalization patient foreign keys",
        )

        for table in TABLES:
            output = pl.scan_parquet(self._output_path(table))
            schema = output.collect_schema()
            if "patient_id" in schema:
                self.assert_zero(
                    _count(output.select("patient_id").join(patients, on="patient_id", how="anti")),
                    f"{table} patient foreign keys",
                )
            if "hospitalization_id" in schema:
                self.assert_zero(
                    _count(
                        output.select("hospitalization_id").join(
                            hospitals.select("hospitalization_id"),
                            on="hospitalization_id",
                            how="anti",
                        )
                    ),
                    f"{table} hospitalization foreign keys",
                )
            if table != "hospitalization" and {
                "patient_id",
                "hospitalization_id",
            }.issubset(schema):
                mismatches = _count(
                    output.select("patient_id", "hospitalization_id")
                    .join(hospitals, on="hospitalization_id", how="left")
                    .filter(
                        ~pl.col("patient_id").eq_missing(
                            pl.col("_hospital_patient_id")
                        )
                    )
                )
                self.assert_zero(mismatches, f"{table} patient/hospital ownership")

    def test_dropped_rows_match_source_relationships_and_rarity(self) -> None:
        patient_schema = pl.scan_parquet(self._source_path("patient")).collect_schema()
        patients = pl.scan_parquet(self._source_path("patient")).select(
            _normalized_identifier(
                "patient_id", patient_schema["patient_id"], "_known_patient"
            ),
            pl.lit(True).alias("_patient_known"),
        )
        hospital_schema = pl.scan_parquet(
            self._source_path("hospitalization")
        ).collect_schema()
        hospitals = (
            pl.scan_parquet(self._source_path("hospitalization"))
            .select(
                _normalized_identifier(
                    "hospitalization_id",
                    hospital_schema["hospitalization_id"],
                    "_known_hospital",
                ),
                _normalized_identifier(
                    "patient_id",
                    hospital_schema["patient_id"],
                    "_hospital_patient",
                ),
                pl.lit(True).alias("_hospital_known"),
            )
            .join(
                patients.select(
                    pl.col("_known_patient").alias("_owner_lookup"),
                    pl.col("_patient_known").alias("_hospital_patient_valid"),
                ),
                left_on="_hospital_patient",
                right_on="_owner_lookup",
                how="left",
            )
        )
        dropped = pl.scan_parquet(self.result.audit_dir / "dropped_rows.parquet")

        diagnosis_source: pl.LazyFrame | None = None
        diagnosis_stray: pl.Expr | None = None
        for table in TABLES:
            schema = pl.scan_parquet(self._source_path(table)).collect_schema()
            source = pl.scan_parquet(self._source_path(table)).with_row_index(
                "source_row_number"
            )
            has_patient = "patient_id" in schema
            has_hospital = "hospitalization_id" in schema
            if has_patient:
                source = source.with_columns(
                    _normalized_identifier(
                        "patient_id", schema["patient_id"], "_source_patient"
                    )
                ).join(
                    patients.select(
                        pl.col("_known_patient").alias("_patient_lookup"),
                        pl.col("_patient_known").alias("_direct_patient_known"),
                    ),
                    left_on="_source_patient",
                    right_on="_patient_lookup",
                    how="left",
                )
            if has_hospital:
                source = source.with_columns(
                    _normalized_identifier(
                        "hospitalization_id",
                        schema["hospitalization_id"],
                        "_source_hospital",
                    )
                ).join(
                    hospitals,
                    left_on="_source_hospital",
                    right_on="_known_hospital",
                    how="left",
                )

            patient_missing = (
                pl.col("_source_patient").is_null()
                if has_patient
                else pl.lit(False)
            )
            patient_unknown = (
                pl.col("_source_patient").is_not_null()
                & pl.col("_direct_patient_known").is_null()
                if has_patient
                else pl.lit(False)
            )
            hospital_missing = (
                pl.col("_source_hospital").is_null()
                if has_hospital
                else pl.lit(False)
            )
            hospital_unknown = (
                pl.col("_source_hospital").is_not_null()
                & pl.col("_hospital_known").is_null()
                if has_hospital
                else pl.lit(False)
            )
            hospital_patient_unknown = (
                pl.col("_hospital_known").is_not_null()
                & pl.col("_hospital_patient_valid").is_null()
                if has_hospital
                else pl.lit(False)
            )
            mismatch = (
                pl.col("_direct_patient_known").is_not_null()
                & pl.col("_hospital_patient_valid").is_not_null()
                & (pl.col("_source_patient") != pl.col("_hospital_patient"))
                if has_patient and has_hospital
                else pl.lit(False)
            )
            stray = (
                patient_missing
                | patient_unknown
                | hospital_missing
                | hospital_unknown
                | hospital_patient_unknown
                | mismatch
            )
            reason = pl.concat_str(
                [
                    pl.when(patient_missing)
                    .then(pl.lit("missing_patient_id"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                    pl.when(patient_unknown)
                    .then(pl.lit("unknown_patient_id"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                    pl.when(hospital_missing)
                    .then(pl.lit("missing_hospitalization_id"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                    pl.when(hospital_unknown)
                    .then(pl.lit("unknown_hospitalization_id"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                    pl.when(hospital_patient_unknown)
                    .then(pl.lit("hospitalization_has_unknown_patient"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                    pl.when(mismatch)
                    .then(pl.lit("patient_hospitalization_mismatch"))
                    .otherwise(pl.lit(None, dtype=pl.String)),
                ],
                separator="|",
                ignore_nulls=True,
            )
            expected = source.filter(stray).select(
                pl.col("source_row_number").cast(pl.UInt64),
                reason.alias("_expected_reason"),
            )
            actual = dropped.filter(
                (pl.col("table") == table)
                & (pl.col("drop_reason") != "rare_diagnosis_code")
            ).select(
                "source_row_number", pl.col("drop_reason").alias("_actual_reason")
            )
            differences = _count(
                expected.join(
                    actual,
                    on="source_row_number",
                    how="full",
                    coalesce=True,
                ).filter(
                    ~pl.col("_expected_reason").eq_missing(pl.col("_actual_reason"))
                )
            )
            self.assert_zero(differences, f"{table} relationship drop audit")

            if table == "hospital_diagnosis":
                diagnosis_source = source
                diagnosis_stray = stray

        if diagnosis_source is None or diagnosis_stray is None:
            self.fail("hospital_diagnosis was not checked")
        valid_diagnoses = diagnosis_source.filter(~diagnosis_stray.fill_null(False))
        diagnosis_frequencies = (
            valid_diagnoses.filter(
                pl.col("diagnosis_code_format").is_not_null()
                & pl.col("diagnosis_code").is_not_null()
            )
            .group_by("diagnosis_code_format", "diagnosis_code")
            .agg(
                pl.col("_source_hospital")
                .n_unique()
                .alias("_hospitalization_count")
            )
        )
        expected_rare = (
            valid_diagnoses.join(
                diagnosis_frequencies,
                on=["diagnosis_code_format", "diagnosis_code"],
                how="left",
            )
            .filter(
                pl.col("_hospitalization_count").is_not_null()
                & (
                    pl.col("_hospitalization_count")
                    < DIAGNOSIS_MIN_HOSPITALIZATIONS
                )
            )
            .select(
                pl.col("source_row_number").cast(pl.UInt64),
                pl.lit(True).alias("_expected"),
            )
        )
        actual_rare = dropped.filter(
            (pl.col("table") == "hospital_diagnosis")
            & (pl.col("drop_reason") == "rare_diagnosis_code")
        ).select("source_row_number", pl.lit(True).alias("_actual"))
        rare_differences = _count(
            expected_rare.join(
                actual_rare,
                on="source_row_number",
                how="full",
                coalesce=True,
            ).filter(~pl.col("_expected").eq_missing(pl.col("_actual")))
        )
        self.assert_zero(rare_differences, "hospital_diagnosis rarity audit")

    def test_exact_identifier_relationships_before_and_after(self) -> None:
        for table in TABLES:
            expected, identifiers, schema = self._expected_identifiers(table)
            if not identifiers:
                continue
            unresolved = pl.any_horizontal(
                [
                    pl.col(f"_old_id_{index}").is_not_null()
                    & pl.col(f"_expected_id_{index}").is_null()
                    for index in range(len(identifiers))
                ]
            )
            self.assert_zero(
                _count(expected.filter(unresolved)), f"{table} unresolved expected IDs"
            )

            expected_counts = (
                expected.select(
                    [
                        pl.col(f"_expected_id_{index}").alias(column)
                        for index, column in enumerate(identifiers)
                    ]
                )
                .group_by(identifiers)
                .len()
                .rename({"len": "_expected_count"})
            )
            actual = pl.scan_parquet(self._output_path(table))
            for column in identifiers:
                self.assertEqual(actual.collect_schema()[column], pl.String, f"{table}.{column}")
            actual_counts = (
                actual.group_by(identifiers)
                .len()
                .rename({"len": "_actual_count"})
            )
            differences = _count(
                expected_counts.join(
                    actual_counts,
                    on=identifiers,
                    how="full",
                    nulls_equal=True,
                    coalesce=True,
                ).filter(
                    pl.col("_expected_count").fill_null(0)
                    != pl.col("_actual_count").fill_null(0)
                )
            )
            self.assert_zero(differences, f"{table} identifier tuple frequencies")

    def test_longitudinal_values_and_offsets(self) -> None:
        patient_map = pl.scan_parquet(
            self.result.audit_dir / "patient_id_mapping.parquet"
        ).select("old_patient_id", "date_shift_days")
        hospital_map = pl.scan_parquet(
            self.result.audit_dir / "hospitalization_id_mapping.parquet"
        ).select("old_hospitalization_id", "old_patient_id")

        for table in TABLES:
            expected, identifiers, schema = self._expected_identifiers(table)
            temporal = [
                name
                for name, dtype in schema.items()
                if dtype == pl.Date or isinstance(dtype, pl.Datetime)
            ]
            if not temporal:
                continue

            if "patient_id" in schema:
                patient_index = identifiers.index("patient_id")
                expected = expected.join(
                    patient_map,
                    left_on=f"_old_id_{patient_index}",
                    right_on="old_patient_id",
                    how="left",
                )
            else:
                hospital_index = identifiers.index("hospitalization_id")
                expected = expected.join(
                    hospital_map,
                    left_on=f"_old_id_{hospital_index}",
                    right_on="old_hospitalization_id",
                    how="left",
                ).join(patient_map, on="old_patient_id", how="left")

            unresolved_dates = pl.any_horizontal(
                [pl.col(column).is_not_null() for column in temporal]
            ) & pl.col("date_shift_days").is_null()
            self.assert_zero(
                _count(expected.filter(unresolved_dates)),
                f"{table} temporal rows without patient offsets",
            )

            transformed: list[pl.Expr] = []
            for column in temporal:
                dtype = schema[column]
                if column == "birth_date":
                    value = pl.lit(None, dtype=dtype)
                else:
                    time_unit = dtype.time_unit if isinstance(dtype, pl.Datetime) else None
                    value = pl.col(column) + pl.duration(
                        days=pl.col("date_shift_days"), time_unit=time_unit
                    )
                    if column == "death_dttm" and isinstance(dtype, pl.Datetime):
                        value = value.dt.truncate("1d")
                transformed.append(value.alias(f"_time_{column}"))

            keys = identifiers
            expected_values = expected.select(
                *[
                    pl.col(f"_expected_id_{index}").alias(column)
                    for index, column in enumerate(identifiers)
                ],
                *transformed,
            )
            actual = pl.scan_parquet(self._output_path(table))
            for column in temporal:
                self.assertEqual(actual.collect_schema()[column], schema[column], f"{table}.{column}")

            def aggregates(prefix: str, columns: list[str]) -> list[pl.Expr]:
                expressions: list[pl.Expr] = []
                for column in columns:
                    expressions.extend(
                        [
                            pl.col(column).null_count().alias(f"{prefix}{column}_nulls"),
                            pl.col(column).min().cast(pl.Int64).alias(f"{prefix}{column}_min"),
                            pl.col(column).max().cast(pl.Int64).alias(f"{prefix}{column}_max"),
                            pl.col(column).cast(pl.Int64).hash(seed=17).sum().alias(
                                f"{prefix}{column}_hash_1"
                            ),
                            pl.col(column).cast(pl.Int64).hash(seed=97).sum().alias(
                                f"{prefix}{column}_hash_2"
                            ),
                        ]
                    )
                return expressions

            expected_time_names = [f"_time_{column}" for column in temporal]
            expected_aggregate = expected_values.group_by(keys).agg(
                aggregates("_expected_", expected_time_names)
            )
            actual_aggregate = actual.group_by(keys).agg(
                aggregates("_actual_", temporal)
            )
            joined = expected_aggregate.join(
                actual_aggregate,
                on=keys,
                how="full",
                nulls_equal=True,
                coalesce=True,
            )
            comparisons: list[pl.Expr] = []
            for source_name, actual_name in zip(expected_time_names, temporal, strict=True):
                for suffix in ("nulls", "min", "max", "hash_1", "hash_2"):
                    comparisons.append(
                        ~pl.col(f"_expected_{source_name}_{suffix}").eq_missing(
                            pl.col(f"_actual_{actual_name}_{suffix}")
                        )
                    )
            differences = _count(joined.filter(pl.any_horizontal(comparisons)))
            self.assert_zero(differences, f"{table} longitudinal aggregates")

    def test_schemas_direct_rules_and_source_immutability(self) -> None:
        for table in TABLES:
            source_path = self._source_path(table)
            output = pl.scan_parquet(self._output_path(table))
            source_schema = pl.scan_parquet(source_path).collect_schema()
            output_schema = output.collect_schema()
            self.assertEqual(list(source_schema), list(output_schema), table)
            for column, source_dtype in source_schema.items():
                expected_dtype = pl.String if column.endswith("_id") else source_dtype
                self.assertEqual(output_schema[column], expected_dtype, f"{table}.{column}")
            for column in GEOCODE_COLUMNS:
                if column in output_schema:
                    self.assert_zero(
                        _count(output.filter(pl.col(column).is_not_null())),
                        f"{table}.{column} retained geocodes",
                    )
            if "birth_date" in output_schema:
                self.assert_zero(
                    _count(output.filter(pl.col("birth_date").is_not_null())),
                    f"{table} retained birth dates",
                )
            if "age_at_admission" in output_schema:
                self.assert_zero(
                    _count(output.filter(pl.col("age_at_admission") > 89)),
                    f"{table} ages above 89",
                )
            if "death_dttm" in output_schema:
                non_midnight = output.filter(
                    pl.col("death_dttm").is_not_null()
                    & (
                        (pl.col("death_dttm").dt.hour() != 0)
                        | (pl.col("death_dttm").dt.minute() != 0)
                        | (pl.col("death_dttm").dt.second() != 0)
                        | (pl.col("death_dttm").dt.nanosecond() != 0)
                    )
                )
                self.assert_zero(_count(non_midnight), f"{table} death times")

            current = source_path.stat()
            self.assertEqual(
                self.source_stats[table],
                (current.st_size, current.st_mtime_ns),
                f"{table} source file changed",
            )


if __name__ == "__main__":
    unittest.main()
