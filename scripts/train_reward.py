from __future__ import annotations

import argparse
import json

from rlhf_pipeline.schemas import PreferencePair
from rlhf_pipeline.training.reward import run_reward_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward model training entrypoint")
    parser.add_argument("--config", required=False, help="Path to production training config")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    result = run_reward_dry_run(
        [
            PreferencePair(
                prompt="Explain PPO.",
                chosen="PPO is a clipped policy optimization algorithm.",
                rejected="PPO is a database.",
            )
        ]
    )
    payload = result.__dict__ | {"config_path": args.config, "mode": "dry-run"}
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

