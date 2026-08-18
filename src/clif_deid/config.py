from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TABLES = (
    "adt",
    "code_status",
    "crrt_therapy",
    "hospital_diagnosis",
    "hospitalization",
    "labs",
    "medication_admin_continuous",
    "medication_admin_intermittent",
    "microbiology_culture",
    "microbiology_susceptibility",
    "patient",
    "patient_assessments",
    "patient_procedures",
    "position",
    "respiratory_support",
    "vitals",
)

GEOCODE_COLUMNS = (
    "zipcode_nine_digit",
    "zipcode_five_digit",
    "census_block_code",
    "census_block_group_code",
    "census_tract",
    "state_code",
    "county_code",
    "fips_version",
)


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


@dataclass(frozen=True)
class Rules:
    replace_patient_id: bool
    replace_hospitalization_id: bool
    replace_other_ids: bool
    remove_stray_ids: bool
    remove_rare_diagnoses: bool
    diagnosis_min_hospitalizations: int
    cap_age_over_89: bool
    null_birth_date: bool
    remove_death_time: bool
    time_shift: bool
    max_offset_days: int
    geocode: dict[str, bool]


@dataclass(frozen=True)
class Config:
    path: Path
    version: str
    input_dir: Path
    output_dir: Path
    non_share_dir: Path
    tables: dict[str, bool]
    rules: Rules

    @property
    def selected_tables(self) -> tuple[str, ...]:
        return tuple(name for name in TABLES if self.tables[name])


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ConfigError("Could not locate the CLIF-DEID project root")


def _check_keys(data: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {location} key(s): {', '.join(unknown)}")


def _flag(value: Any, location: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ConfigError(f"{location} must be the integer 0 or 1")
    return bool(value)


def _required(data: dict[str, Any], key: str, location: str) -> Any:
    if key not in data:
        raise ConfigError(f"Missing required {location} key: {key}")
    return data[key]


def _resolve_path(value: Any, config_dir: Path, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    _check_keys(
        raw,
        {"version", "input_dir", "output_dir", "non_share_dir", "tables", "rules"},
        "top-level",
    )

    version = str(_required(raw, "version", "top-level"))
    if version == "3.0":
        raise ConfigError(
            "CLIF 3.0 is recognized but not supported until its finalized schema is supplied"
        )
    if version != "2.1":
        raise ConfigError("version must be \"2.1\" or \"3.0\"")

    config_dir = config_path.parent
    root = _project_root()
    input_dir = _resolve_path(
        _required(raw, "input_dir", "top-level"), config_dir, "input_dir"
    )
    output_dir = (
        _resolve_path(raw["output_dir"], config_dir, "output_dir")
        if "output_dir" in raw
        else (root / "De-id").resolve()
    )
    non_share_dir = (
        _resolve_path(raw["non_share_dir"], config_dir, "non_share_dir")
        if "non_share_dir" in raw
        else (root / "Non-Share").resolve()
    )

    table_raw = _required(raw, "tables", "top-level")
    if not isinstance(table_raw, dict):
        raise ConfigError("tables must be a TOML table")
    _check_keys(table_raw, set(TABLES), "tables")
    missing_tables = [name for name in TABLES if name not in table_raw]
    if missing_tables:
        raise ConfigError(
            "Missing table selection flag(s): " + ", ".join(missing_tables)
        )
    tables = {name: _flag(table_raw[name], f"tables.{name}") for name in TABLES}
    if not any(tables.values()):
        raise ConfigError("At least one table must be selected")

    rules_raw = _required(raw, "rules", "top-level")
    if not isinstance(rules_raw, dict):
        raise ConfigError("rules must be a TOML table")
    rule_names = {
        "replace_patient_id",
        "replace_hospitalization_id",
        "replace_other_ids",
        "remove_stray_ids",
        "remove_rare_diagnoses",
        "diagnosis_min_hospitalizations",
        "cap_age_over_89",
        "null_birth_date",
        "remove_death_time",
        "time_shift",
        "max_offset_days",
        "geocode",
    }
    _check_keys(rules_raw, rule_names, "rules")
    missing_rules = sorted(rule_names - set(rules_raw))
    if missing_rules:
        raise ConfigError("Missing rule(s): " + ", ".join(missing_rules))

    minimum = rules_raw["diagnosis_min_hospitalizations"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ConfigError("rules.diagnosis_min_hospitalizations must be at least 1")
    maximum = rules_raw["max_offset_days"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ConfigError("rules.max_offset_days must be a non-negative integer")

    geocode_raw = rules_raw["geocode"]
    if not isinstance(geocode_raw, dict):
        raise ConfigError("rules.geocode must be a TOML table")
    _check_keys(geocode_raw, set(GEOCODE_COLUMNS), "rules.geocode")
    missing_geocodes = [name for name in GEOCODE_COLUMNS if name not in geocode_raw]
    if missing_geocodes:
        raise ConfigError(
            "Missing geocode flag(s): " + ", ".join(missing_geocodes)
        )

    rules = Rules(
        replace_patient_id=_flag(
            rules_raw["replace_patient_id"], "rules.replace_patient_id"
        ),
        replace_hospitalization_id=_flag(
            rules_raw["replace_hospitalization_id"],
            "rules.replace_hospitalization_id",
        ),
        replace_other_ids=_flag(
            rules_raw["replace_other_ids"], "rules.replace_other_ids"
        ),
        remove_stray_ids=_flag(
            rules_raw["remove_stray_ids"], "rules.remove_stray_ids"
        ),
        remove_rare_diagnoses=_flag(
            rules_raw["remove_rare_diagnoses"], "rules.remove_rare_diagnoses"
        ),
        diagnosis_min_hospitalizations=minimum,
        cap_age_over_89=_flag(
            rules_raw["cap_age_over_89"], "rules.cap_age_over_89"
        ),
        null_birth_date=_flag(
            rules_raw["null_birth_date"], "rules.null_birth_date"
        ),
        remove_death_time=_flag(
            rules_raw["remove_death_time"], "rules.remove_death_time"
        ),
        time_shift=_flag(rules_raw["time_shift"], "rules.time_shift"),
        max_offset_days=maximum,
        geocode={
            name: _flag(geocode_raw[name], f"rules.geocode.{name}")
            for name in GEOCODE_COLUMNS
        },
    )
    return Config(
        path=config_path,
        version=version,
        input_dir=input_dir,
        output_dir=output_dir,
        non_share_dir=non_share_dir,
        tables=tables,
        rules=rules,
    )
