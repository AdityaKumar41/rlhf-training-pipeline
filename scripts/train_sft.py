from __future__ import annotations

import argparse
import json

from rlhf_pipeline.training.sft import run_sft_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training entrypoint")
    parser.add_argument("--config", required=False, help="Path to production training config")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    result = run_sft_dry_run(
        [("Explain RLHF.", "RLHF aligns model outputs with human preferences.")]
    )
    payload = result.__dict__ | {"config_path": args.config, "mode": "dry-run"}
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

