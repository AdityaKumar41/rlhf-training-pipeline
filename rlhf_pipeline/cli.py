from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlhf_pipeline.curation.normalize import load_jsonl, write_sharegpt_jsonl
from rlhf_pipeline.curation.pipeline import curate_records
from rlhf_pipeline.schemas import PreferencePair
from rlhf_pipeline.training.ppo import run_ppo_dry_run
from rlhf_pipeline.training.reward import run_reward_dry_run
from rlhf_pipeline.training.sft import run_sft_dry_run


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


def curate_command(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.input))
    conversations, report = curate_records(records, source=args.source, target_size=args.target_size)
    write_sharegpt_jsonl(Path(args.output), conversations)
    _print_json(report.to_dict())


def sft_dry_run_command(_: argparse.Namespace) -> None:
    result = run_sft_dry_run(
        [
            ("Explain KL regularization in PPO.", "KL keeps the policy close to a reference model."),
            ("What is a reward model?", "A reward model scores candidate responses from preferences."),
        ]
    )
    _print_json(result.__dict__)


def reward_dry_run_command(_: argparse.Namespace) -> None:
    pairs = [
        PreferencePair(
            prompt="Explain RLHF simply.",
            chosen="RLHF trains a model using human preference signals.",
            rejected="It is a thing with computers.",
        )
    ]
    result = run_reward_dry_run(pairs)
    _print_json(result.__dict__)


def ppo_dry_run_command(_: argparse.Namespace) -> None:
    result = run_ppo_dry_run(
        responses=[
            "RLHF combines supervised tuning, reward modeling, and PPO.",
            "The KL penalty limits drift from the reference model.",
        ],
        rewards=[1.2, 0.7],
        running_kl=0.02,
    )
    _print_json(result.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLHF training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    curate = subparsers.add_parser("curate", help="Normalize, dedupe, safety-filter, and sample JSONL data")
    curate.add_argument("--input", required=True)
    curate.add_argument("--output", required=True)
    curate.add_argument("--source", default="local")
    curate.add_argument("--target-size", type=int, default=None)
    curate.set_defaults(func=curate_command)

    sft = subparsers.add_parser("sft-dry-run", help="Validate SFT configuration and dataset stats")
    sft.set_defaults(func=sft_dry_run_command)

    reward = subparsers.add_parser("reward-dry-run", help="Validate reward model data contracts")
    reward.set_defaults(func=reward_dry_run_command)

    ppo = subparsers.add_parser("ppo-dry-run", help="Validate PPO metrics and KL controller")
    ppo.set_defaults(func=ppo_dry_run_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

