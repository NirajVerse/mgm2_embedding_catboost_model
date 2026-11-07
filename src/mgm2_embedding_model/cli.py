"""Command-line interface for preparation and paired CatBoost modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from .config import load_config
from .model_catboost import CatBoostDependencyError
from .model_config import apply_runtime_overrides, load_model_config
from .model_pipeline import modeling_preflight, run_modeling
from .prepare import prepare, with_output_override
from .validation import ValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mgm2-embedding-data",
        description=(
            "Validate and prepare paired MGM2 embeddings, then compare "
            "temperature models on frozen cruise-grouped splits."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "prepare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            required=True,
            help="Path to the JSON preparation configuration.",
        )
        if command == "prepare":
            subparser.add_argument(
                "--output-dir",
                type=Path,
                help="Optional output directory override (relative to project root).",
            )
    preflight = subparsers.add_parser(
        "model-preflight",
        help="Validate all model inputs and the frozen ASV reference without training.",
    )
    preflight.add_argument(
        "--config", required=True, help="Path to the JSON modeling configuration."
    )

    train = subparsers.add_parser(
        "train",
        help="Tune and evaluate paired original/MAPseq CatBoost temperature models.",
    )
    train.add_argument(
        "--config", required=True, help="Path to the JSON modeling configuration."
    )
    train.add_argument(
        "--run-id",
        required=True,
        help="New results-directory name; an existing nonempty run is never overwritten.",
    )
    train.add_argument(
        "--task-type",
        choices=("CPU", "GPU", "cpu", "gpu"),
        help="Optional CatBoost execution override (default comes from the config).",
    )
    train.add_argument(
        "--devices",
        help="Optional CatBoost GPU device string, for example '0'.",
    )
    train.add_argument(
        "--thread-count",
        type=int,
        help="Optional CatBoost thread-count override.",
    )
    train.add_argument(
        "--verbose",
        type=int,
        help="CatBoost logging interval; use 0 to suppress iteration logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"model-preflight", "train"}:
            model_config = load_model_config(args.config)
            if args.command == "model-preflight":
                result = modeling_preflight(model_config)
                print(json.dumps(result, indent=2, sort_keys=True))
                print("Preflight passed. No CatBoost model was trained.")
                return 0
            model_config = apply_runtime_overrides(
                model_config,
                task_type=args.task_type,
                devices=args.devices,
                thread_count=args.thread_count,
                verbose=args.verbose,
            )
            result = run_modeling(model_config, args.run_id)
            print(json.dumps(result, indent=2, sort_keys=True))
            print(f"Results: {model_config.results_dir / args.run_id}")
            return 0

        config = load_config(args.config)
        if args.command == "validate":
            # Exercise the complete preparation in a disposable directory so the
            # command validates joins, averaging, targets, splits, and output schemas.
            with tempfile.TemporaryDirectory(prefix="mgm2_embedding_validation_") as temporary:
                validation_config = with_output_override(config, Path(temporary))
                result = prepare(validation_config)
            print(json.dumps(result["summary"], indent=2))
            return 0
        config = with_output_override(config, args.output_dir)
        result = prepare(config)
        print(json.dumps(result["summary"], indent=2))
        print(f"Prepared files: {result['output_dir']}")
        return 0
    except ValidationError as error:
        print(str(error), file=sys.stderr)
        for failure in error.recorder.failures:
            print(
                f"FAIL {failure['check']}: observed={failure['observed']} "
                f"expected={failure['expected']}",
                file=sys.stderr,
            )
        return 2
    except CatBoostDependencyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
