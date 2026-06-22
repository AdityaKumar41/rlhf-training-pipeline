from __future__ import annotations

import argparse
import json

from rlhf_pipeline.training.ppo import run_ppo_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training entrypoint")
    parser.add_argument("--config", required=False, help="Path to production training config")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    result = run_ppo_dry_run(
        responses=[
            "PPO improves the SFT policy using reward scores.",
            "Adaptive KL control reduces reward hacking risk.",
        ],
        rewards=[1.0, 0.8],
        running_kl=0.012,
    )
    payload = result.__dict__ | {"config_path": args.config, "mode": "dry-run"}
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

