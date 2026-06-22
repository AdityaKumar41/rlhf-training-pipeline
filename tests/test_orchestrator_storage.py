from rlhf_pipeline.orchestrator.jobs import JobSpec, estimate_job_cost, render_submission_command
from rlhf_pipeline.schemas import Stage
from rlhf_pipeline.storage.registry import ArtifactManifest, LocalArtifactRegistry


def test_render_local_and_slurm_commands() -> None:
    local = JobSpec(name="local-sft", stage=Stage.SFT, command=["python", "scripts/train_sft.py"])
    assert render_submission_command(local) == ["python", "scripts/train_sft.py"]

    slurm = JobSpec(
        name="slurm-sft",
        stage=Stage.SFT,
        command=["python", "scripts/train_sft.py"],
        backend="slurm",
    )
    command = render_submission_command(slurm)
    assert command[0] == "sbatch"
    assert "--gres" in command


def test_estimate_job_cost_spot_discount() -> None:
    assert estimate_job_cost(hours=10, gpu_count=4, hourly_gpu_price=5, spot=False) == 200
    assert estimate_job_cost(hours=10, gpu_count=4, hourly_gpu_price=5, spot=True) == 70


def test_artifact_manifest_roundtrip(tmp_path) -> None:
    registry = LocalArtifactRegistry(tmp_path)
    manifest = ArtifactManifest(
        run_id="run-1",
        stage=Stage.SFT,
        artifact_type="lora-adapter",
        files=["adapter.safetensors"],
        metrics={"validation_loss": 1.2},
    )
    registry.write_manifest(manifest)
    loaded = registry.load_manifest(Stage.SFT, "run-1", "lora-adapter")
    assert loaded.run_id == manifest.run_id
    assert loaded.stage == Stage.SFT
    assert loaded.metrics["validation_loss"] == 1.2

