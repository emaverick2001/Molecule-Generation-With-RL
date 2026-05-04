import json
from pathlib import Path

import yaml

from src.pipeline.run_posttraining import run_posttraining


def _sdf(atom_coordinates):
    atom_lines = "\n".join(
        f"{x:10.4f}{y:10.4f}{z:10.4f} {atom:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for atom, x, y, z in atom_coordinates
    )
    return f"""TestMol
  MVP

{len(atom_coordinates):3d}  0  0  0  0  0            999 V2000
{atom_lines}
M  END
$$$$
"""


def test_run_posttraining_offline_reward_debug_creates_artifacts(tmp_path):
    source_run = tmp_path / "source_run"
    source_run.mkdir()
    pose = source_run / "pose.sdf"
    reference = source_run / "ligand_gt.sdf"
    text = _sdf([("C", 0.0, 0.0, 0.0)])
    pose.write_text(text, encoding="utf-8")
    reference.write_text(text, encoding="utf-8")

    (source_run / "input_manifest.json").write_text(
        json.dumps(
            [
                {
                    "complex_id": "1abc",
                    "protein_path": str(source_run / "protein.pdb"),
                    "ligand_path": str(source_run / "ligand.sdf"),
                    "ground_truth_pose_path": str(reference),
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )
    (source_run / "generated_samples_manifest.json").write_text(
        json.dumps(
            [
                {
                    "complex_id": "1abc",
                    "sample_id": 0,
                    "pose_path": str(pose),
                    "confidence_score": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "rl.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "test",
                    "model": "diffdock",
                    "mode": "posttraining_offline_reward_debug",
                    "seed": 7,
                },
                "algorithm": {"name": "offline_reward_debug"},
                "data": {"source_run_dir": str(source_run)},
                "reward": {"weights": {"rmsd": 1.0}},
                "rollout": {"min_valid_samples_per_complex": 1},
                "artifacts": {
                    "run_root": str(tmp_path / "runs"),
                    "run_tag": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = run_posttraining(config_path)

    assert (run_dir / "config_snapshot.json").is_file()
    assert (run_dir / "rollouts" / "offline" / "rollout.jsonl").is_file()
    assert (run_dir / "rollouts" / "offline" / "rewards.csv").is_file()
    summary = json.loads((run_dir / "posttraining_summary.json").read_text())
    assert summary["num_rollout_records"] == 1


def test_run_posttraining_grpo_surrogate_creates_checkpoint(tmp_path):
    source_run = tmp_path / "source_run"
    source_run.mkdir()
    reference = source_run / "ligand_gt.sdf"
    reference.write_text(_sdf([("C", 0.0, 0.0, 0.0)]), encoding="utf-8")

    generated = []
    for sample_id, x in enumerate([0.0, 0.3, 0.8, 1.2]):
        pose = source_run / f"pose_{sample_id}.sdf"
        pose.write_text(_sdf([("C", x, 0.0, 0.0)]), encoding="utf-8")
        generated.append(
            {
                "complex_id": "1abc",
                "sample_id": sample_id,
                "pose_path": str(pose),
                "confidence_score": 1.0 - sample_id * 0.1,
            }
        )

    (source_run / "input_manifest.json").write_text(
        json.dumps(
            [
                {
                    "complex_id": "1abc",
                    "protein_path": str(source_run / "protein.pdb"),
                    "ligand_path": str(source_run / "ligand.sdf"),
                    "ground_truth_pose_path": str(reference),
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )
    (source_run / "generated_samples_manifest.json").write_text(
        json.dumps(generated),
        encoding="utf-8",
    )
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "test",
                    "model": "diffdock",
                    "mode": "posttraining_grpo_surrogate_smoke",
                    "seed": 7,
                },
                "algorithm": {
                    "name": "grpo_surrogate",
                    "surrogate_backend": "debug_linear",
                    "learning_rate": 0.05,
                    "grpo_epochs": 1,
                },
                "data": {"source_run_dir": str(source_run)},
                "reward": {"weights": {"rmsd": 1.0}},
                "rollout": {
                    "samples_per_complex": 4,
                    "min_valid_samples_per_complex": 2,
                },
                "artifacts": {
                    "run_root": str(tmp_path / "runs"),
                    "run_tag": "grpo-unit",
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = run_posttraining(config_path)

    checkpoint = run_dir / "checkpoints" / "grpo_debug_linear_step1.json"
    summary = json.loads((run_dir / "posttraining_summary.json").read_text())
    grpo_summary = summary["metrics"]["training"]

    assert checkpoint.is_file()
    assert (run_dir / "rollouts" / "grpo_step_000" / "surrogate_scores.csv").is_file()
    assert summary["algorithm"] == "grpo_surrogate"
    assert grpo_summary["final_loss"] < grpo_summary["initial_loss"]
