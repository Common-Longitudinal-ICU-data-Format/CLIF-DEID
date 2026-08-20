from __future__ import annotations

import argparse
import sys

from clif_deid.config import ConfigError, load_config
from clif_deid.pipeline import DeidentificationError, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clif-deid",
        description="De-identify selected CLIF Parquet tables",
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
        result = run(config, overwrite=args.overwrite)
    except (ConfigError, DeidentificationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"De-identified tables: {result.output_dir}")
    print(f"Private audit: {result.audit_dir}")
