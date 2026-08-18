from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import polars as pl

from clif_deid.config import ConfigError, GEOCODE_COLUMNS, TABLES, load_config
from clif_deid.pipeline import DeidentificationError, run


def _write_config(
    root: Path,
    selected: set[str],
    *,
    version: str = "2.1",
    remove_stray_ids: int = 1,
    time_shift: int = 1,
) -> Path:
    table_lines = "\n".join(
        f"{table} = {1 if table in selected else 0}" for table in TABLES
    )
    geocode_lines = "\n".join(f"{column} = 1" for column in GEOCODE_COLUMNS)
    text = f'''version = "{version}"
input_dir = "{root / 'input'}"
output_dir = "{root / 'output'}"
non_share_dir = "{root / 'private'}"

[tables]
{table_lines}

[rules]
replace_patient_id = 1
replace_hospitalization_id = 1
replace_other_ids = 1
remove_stray_ids = {remove_stray_ids}
remove_rare_diagnoses = 1
diagnosis_min_hospitalizations = 2
cap_age_over_89 = 1
null_birth_date = 1
remove_death_time = 1
time_shift = {time_shift}
max_offset_days = 45

[rules.geocode]
{geocode_lines}
'''
    path = root / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _write_anchors(input_dir: Path) -> None:
    death = pl.Series(
        "death_dttm",
        [datetime(2024, 1, 20, 12, 34, 56), None, None],
        dtype=pl.Datetime("ns", "America/New_York"),
    )
    pl.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "birth_date": [
                date(1930, 1, 1),
                date(1980, 2, 2),
                date(1990, 3, 3),
            ],
        }
    ).with_columns(death).write_parquet(input_dir / "clif_patient.parquet")

    admission = pl.Series(
        "admission_dttm",
        [
            datetime(2024, 1, 1, 8),
            datetime(2024, 2, 1, 8),
            datetime(2024, 1, 3, 8),
            datetime(2024, 1, 4, 8),
        ],
        dtype=pl.Datetime("us", "America/New_York"),
    )
    pl.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "hospitalization_id": ["h1", "h2", "h3", "h4"],
            "hospitalization_joined_id": ["j1", "j1", "j3", "j4"],
            "age_at_admission": [94, 94, 44, 33],
            "zipcode_nine_digit": ["123456789"] * 4,
            "zipcode_five_digit": ["12345"] * 4,
            "census_block_code": ["block"] * 4,
            "census_block_group_code": ["group"] * 4,
            "census_tract": ["tract"] * 4,
            "state_code": ["12"] * 4,
            "county_code": ["12345"] * 4,
            "fips_version": ["2020"] * 4,
        }
    ).with_columns(admission).write_parquet(
        input_dir / "clif_hospitalization.parquet"
    )


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_and_private_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_anchors(input_dir)

            pl.DataFrame(
                {
                    "hospitalization_id": ["h1", "h3", "h1", "h1", "missing"],
                    "diagnosis_code_format": ["ICD10CM"] * 5,
                    "diagnosis_code": ["A1", "A1", "RARE", "RARE", "A1"],
                    "diagnosis_primary": [1, 0, 0, 0, 0],
                }
            ).write_parquet(input_dir / "clif_hospital_diagnosis.parquet")

            lab_times = pl.Series(
                "lab_result_dttm",
                [datetime(2024, 1, 2, 10), datetime(2024, 2, 2, 10)],
                dtype=pl.Datetime("ns", "America/New_York"),
            )
            pl.DataFrame(
                {
                    "hospitalization_id": ["h1", "h2"],
                    "lab_name": ["x", "x"],
                }
            ).with_columns(lab_times).write_parquet(input_dir / "clif_labs.parquet")

            pl.DataFrame(
                {
                    "patient_id": ["p1", "p2", "unknown"],
                    "hospitalization_id": ["h1", "h1", "h1"],
                    "organism_id": ["o1", "o2", "o3"],
                }
            ).write_parquet(input_dir / "clif_microbiology_culture.parquet")

            selected = {
                "patient",
                "hospitalization",
                "hospital_diagnosis",
                "labs",
                "microbiology_culture",
            }
            config = load_config(_write_config(root, selected))
            result = run(config)

            patient_map = pl.read_parquet(
                result.audit_dir / "patient_id_mapping.parquet"
            )
            hospital_map = pl.read_parquet(
                result.audit_dir / "hospitalization_id_mapping.parquet"
            )
            self.assertEqual(patient_map["new_patient_id"].n_unique(), 3)
            self.assertTrue(
                patient_map["new_patient_id"]
                .str.contains(r"^[0-9a-f]{32}$")
                .all()
            )
            self.assertEqual(hospital_map["new_hospitalization_id"].n_unique(), 4)

            output_patient = pl.read_parquet(result.output_dir / "clif_patient.parquet")
            p1 = patient_map.filter(pl.col("old_patient_id") == "p1").row(
                0, named=True
            )
            self.assertIn(p1["new_patient_id"], output_patient["patient_id"].to_list())
            output_p1 = output_patient.filter(
                pl.col("patient_id") == p1["new_patient_id"]
            ).row(0, named=True)
            self.assertEqual(output_patient["birth_date"].null_count(), 3)
            self.assertEqual(
                output_patient.schema["death_dttm"],
                pl.Datetime("ns", "America/New_York"),
            )
            shifted_death = output_p1["death_dttm"]
            self.assertEqual(
                (shifted_death.hour, shifted_death.minute, shifted_death.second),
                (0, 0, 0),
            )
            self.assertEqual(
                (shifted_death.date() - date(2024, 1, 20)).days,
                p1["date_shift_days"],
            )

            output_hospital = pl.read_parquet(
                result.output_dir / "clif_hospitalization.parquet"
            )
            self.assertEqual(output_hospital["age_at_admission"].max(), 89)
            self.assertEqual(output_hospital["hospitalization_joined_id"].null_count(), 0)
            self.assertTrue(
                output_hospital["hospitalization_joined_id"]
                .str.contains(r"^[0-9a-f]{32}$")
                .all()
            )
            self.assertEqual(
                output_hospital.filter(pl.col("patient_id") == p1["new_patient_id"])[
                    "hospitalization_joined_id"
                ].n_unique(),
                1,
            )
            for column in GEOCODE_COLUMNS:
                self.assertEqual(output_hospital[column].null_count(), 4)
            self.assertEqual(
                output_hospital.schema["admission_dttm"],
                pl.Datetime("us", "America/New_York"),
            )

            output_labs = pl.read_parquet(result.output_dir / "clif_labs.parquet")
            self.assertEqual(
                output_labs.schema["lab_result_dttm"],
                pl.Datetime("ns", "America/New_York"),
            )
            original_labs = pl.read_parquet(input_dir / "clif_labs.parquet")
            hospital_lookup = {
                row["new_hospitalization_id"]: row["old_hospitalization_id"]
                for row in hospital_map.iter_rows(named=True)
            }
            original_lookup = {
                row["hospitalization_id"]: row["lab_result_dttm"]
                for row in original_labs.iter_rows(named=True)
            }
            deltas = []
            for row in output_labs.iter_rows(named=True):
                old_id = hospital_lookup[row["hospitalization_id"]]
                delta = row["lab_result_dttm"] - original_lookup[old_id]
                deltas.append(int(delta.total_seconds() // 86400))
            self.assertEqual(deltas, [p1["date_shift_days"], p1["date_shift_days"]])

            diagnoses = pl.read_parquet(
                result.output_dir / "clif_hospital_diagnosis.parquet"
            )
            self.assertEqual(diagnoses.height, 2)
            self.assertEqual(diagnoses["diagnosis_code"].to_list(), ["A1", "A1"])

            cultures = pl.read_parquet(
                result.output_dir / "clif_microbiology_culture.parquet"
            )
            self.assertEqual(cultures.height, 1)
            self.assertEqual(cultures["patient_id"][0], p1["new_patient_id"])

            dropped = pl.read_parquet(result.audit_dir / "dropped_rows.parquet")
            self.assertEqual(dropped.height, 5)
            reasons = "|".join(dropped["drop_reason"].to_list())
            self.assertIn("unknown_hospitalization_id", reasons)
            self.assertIn("patient_hospitalization_mismatch", reasons)
            self.assertIn("unknown_patient_id", reasons)
            self.assertIn("rare_diagnosis_code", reasons)
            self.assertNotIn("lab_name", dropped.columns)

            report = json.loads(
                (result.audit_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["tables"]["hospital_diagnosis"]["output_rows"], 2)
            self.assertEqual(
                report["tables"]["hospital_diagnosis"][
                    "rare_diagnosis_rows_dropped"
                ],
                2,
            )
            self.assertEqual(report["drop_reasons"]["rare_diagnosis_code"], 2)
            self.assertEqual(
                report["drop_reasons"]["patient_hospitalization_mismatch"], 1
            )
            self.assertEqual(report["totals"]["birth_dates_nulled"], 3)
            self.assertEqual(report["totals"]["death_datetimes_truncated"], 1)

    def test_every_id_domain_is_rekeyed_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_anchors(input_dir)
            pl.DataFrame(
                {
                    "hospitalization_id": ["h1"],
                    "hospital_id": ["hospital-A"],
                }
            ).write_parquet(input_dir / "clif_adt.parquet")
            pl.DataFrame(
                {"patient_id": ["p1"], "rush_proc_id": ["rush-1"]}
            ).write_parquet(input_dir / "clif_code_status.parquet")
            pl.DataFrame(
                {
                    "hospitalization_id": ["h1", "h2"],
                    "device_id": ["device-1", None],
                }
            ).write_parquet(input_dir / "clif_crrt_therapy.parquet")
            for table in (
                "medication_admin_continuous",
                "medication_admin_intermittent",
            ):
                pl.DataFrame(
                    {"hospitalization_id": ["h1"], "med_order_id": ["med-1"]}
                ).write_parquet(input_dir / f"clif_{table}.parquet")
            pl.DataFrame(
                {
                    "patient_id": ["p1"],
                    "hospitalization_id": ["h1"],
                    "organism_id": ["organism-1"],
                }
            ).write_parquet(input_dir / "clif_microbiology_culture.parquet")
            pl.DataFrame({"organism_id": ["organism-1"]}).write_parquet(
                input_dir / "clif_microbiology_susceptibility.parquet"
            )
            pl.DataFrame(
                {
                    "patient_id": ["p1"],
                    "hospitalization_id": ["h1"],
                    "billing_provider_id": ["provider-1"],
                    "performing_provider_id": ["provider-1"],
                }
            ).write_parquet(input_dir / "clif_patient_procedures.parquet")
            pl.DataFrame(
                {
                    "hospitalization_id": ["h1"],
                    "device_id": ["device-1"],
                }
            ).write_parquet(input_dir / "clif_respiratory_support.parquet")

            selected = {
                "adt",
                "code_status",
                "crrt_therapy",
                "hospitalization",
                "medication_admin_continuous",
                "medication_admin_intermittent",
                "microbiology_culture",
                "microbiology_susceptibility",
                "patient",
                "patient_procedures",
                "respiratory_support",
            }
            result = run(load_config(_write_config(root, selected)))

            expected_mapping_files = {
                "patient_id_mapping.parquet",
                "hospitalization_id_mapping.parquet",
                "hospitalization_joined_id_mapping.parquet",
                "hospital_id_mapping.parquet",
                "rush_proc_id_mapping.parquet",
                "device_id_mapping.parquet",
                "med_order_id_mapping.parquet",
                "organism_id_mapping.parquet",
                "provider_id_mapping.parquet",
            }
            self.assertTrue(
                expected_mapping_files.issubset(
                    {path.name for path in result.audit_dir.glob("*.parquet")}
                )
            )

            crrt = pl.read_parquet(result.output_dir / "clif_crrt_therapy.parquet")
            respiratory = pl.read_parquet(
                result.output_dir / "clif_respiratory_support.parquet"
            )
            self.assertEqual(crrt["device_id"].null_count(), 1)
            self.assertEqual(crrt["device_id"].drop_nulls()[0], respiratory["device_id"][0])

            continuous = pl.read_parquet(
                result.output_dir / "clif_medication_admin_continuous.parquet"
            )
            intermittent = pl.read_parquet(
                result.output_dir / "clif_medication_admin_intermittent.parquet"
            )
            self.assertEqual(continuous["med_order_id"][0], intermittent["med_order_id"][0])

            culture = pl.read_parquet(
                result.output_dir / "clif_microbiology_culture.parquet"
            )
            susceptibility = pl.read_parquet(
                result.output_dir / "clif_microbiology_susceptibility.parquet"
            )
            self.assertEqual(culture["organism_id"][0], susceptibility["organism_id"][0])

            procedures = pl.read_parquet(
                result.output_dir / "clif_patient_procedures.parquet"
            )
            self.assertEqual(
                procedures["billing_provider_id"][0],
                procedures["performing_provider_id"][0],
            )

            for output_path in result.output_dir.glob("*.parquet"):
                frame = pl.read_parquet(output_path)
                for column in frame.columns:
                    if column.endswith("_id"):
                        values = frame[column].drop_nulls().cast(pl.String)
                        self.assertTrue(
                            values.str.contains(r"^[0-9a-f]{32}$").all(),
                            f"{output_path.name}.{column} was not fully rekeyed",
                        )

            for mapping_name in expected_mapping_files - {
                "patient_id_mapping.parquet",
                "hospitalization_id_mapping.parquet",
            }:
                mapping = pl.read_parquet(result.audit_dir / mapping_name)
                self.assertTrue(
                    set(mapping["old_id"].drop_nulls()).isdisjoint(
                        set(mapping["new_id"].drop_nulls())
                    )
                )

    def test_unresolved_dates_fail_without_stray_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_anchors(input_dir)
            pl.DataFrame(
                {
                    "hospitalization_id": ["unknown"],
                    "lab_result_dttm": [datetime(2024, 1, 1)],
                }
            ).write_parquet(input_dir / "clif_labs.parquet")
            config = load_config(
                _write_config(root, {"labs"}, remove_stray_ids=0)
            )
            with self.assertRaisesRegex(DeidentificationError, "enable remove_stray_ids"):
                run(config)

    def test_version_3_is_recognized_but_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_config(root, {"patient"}, version="3.0")
            with self.assertRaisesRegex(ConfigError, "recognized but not supported"):
                load_config(path)

    def test_selected_table_requires_its_clif_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_anchors(input_dir)
            pl.DataFrame({"lab_name": ["x"]}).write_parquet(
                input_dir / "clif_labs.parquet"
            )
            config = load_config(_write_config(root, {"labs"}))
            with self.assertRaisesRegex(DeidentificationError, "hospitalization_id"):
                run(config)

    def test_unknown_id_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            _write_anchors(input_dir)
            patient = pl.read_parquet(input_dir / "clif_patient.parquet").with_columns(
                pl.lit("source-value").alias("unregistered_id")
            )
            patient.write_parquet(input_dir / "clif_patient.parquet")
            config = load_config(_write_config(root, {"patient"}))
            with self.assertRaisesRegex(DeidentificationError, "unrecognized ID column"):
                run(config)

    def test_boolean_is_not_accepted_as_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_config(root, {"patient"})
            text = path.read_text(encoding="utf-8").replace("patient = 1", "patient = true")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "integer 0 or 1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
