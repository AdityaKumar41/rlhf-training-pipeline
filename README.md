# RLHF Training Pipeline

This repository implements the product described in `prd-rlhf-training-pipeline.docx`: a modular RLHF platform covering data curation, SFT, reward modeling, PPO, annotation collection, evaluation metrics, orchestration, and artifact manifests.

The heavy training stack is isolated behind clean Python modules so the repo works in two modes:

- Local mode: deterministic dry-runs, curation, annotation API, tests, and artifact manifests on a laptop.
- GPU mode: install the full dependencies from `pyproject.toml`, wire Hugging Face/TRL/DeepSpeed/W&B/S3 credentials, and submit jobs from `configs/jobs`.

## Quick Start

```bash
python3 -m pytest
python3 -m rlhf_pipeline.cli sft-dry-run
python3 -m rlhf_pipeline.cli reward-dry-run
python3 -m rlhf_pipeline.cli ppo-dry-run
python3 -m rlhf_pipeline.cli curate --input sample_data/raw_instructions.jsonl --output tmp/curated.jsonl --target-size 2
```

Run the annotation API after installing dependencies:

```bash
python3 -m pip install -e ".[dev]"
uvicorn rlhf_pipeline.api.main:app --reload --port 8000
```

Run the Next.js annotation UI:

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` if the API runs somewhere else.

## What Is Implemented

- Data curation pipeline: ShareGPT normalization, near-deduplication, safety filtering, tempered diversity sampling, and report generation.
- Training contracts: SFT config, response-only loss masks, LoRA parameter estimates, reward model Bradley-Terry math, PPO reward normalization, adaptive KL controller, and dry-run entrypoints.
- Annotation platform backend: FastAPI endpoints with SQLite persistence for task queues, annotation events, skip/tie choices, and stats.
- Annotation platform frontend: keyboard-friendly task review screen with side-by-side candidates and live progress stats.
- Evaluation utilities: distinct n-gram diversity, Cohen's kappa, calibration bins, and length-bias deltas.
- Orchestration utilities: local, Slurm, and Kubernetes command rendering with simple GPU cost estimates.
- Artifact registry: local model/artifact manifest format mirroring the PRD's S3 model registry layout.
- Config examples: DeepSpeed ZeRO-3 and stage job specs.

## Repository Layout

```text
rlhf_pipeline/
  api/             FastAPI annotation backend
  curation/        normalization, dedupe, safety, diversity sampling
  eval/            quality and bias metrics
  orchestrator/    job specs and submission command rendering
  storage/         artifact manifest registry
  training/        SFT, reward, and PPO config/math/dry-runs
frontend/          Next.js annotation UI
configs/           DeepSpeed and job examples
docs/              architecture, operations, and model-card templates
tests/             local verification suite
sample_data/       tiny local dataset for smoke tests
```

## Production Notes

The PRD targets Llama-3-8B with LoRA, TRL trainers, DeepSpeed ZeRO-3, W&B, S3, vLLM, and PostgreSQL. This repo deliberately keeps those integrations explicit but not mandatory for local tests. The next production step is replacing dry-run entrypoints with the actual TRL `SFTTrainer`, `RewardTrainer`, and `PPOTrainer` calls using the configs already represented here.

