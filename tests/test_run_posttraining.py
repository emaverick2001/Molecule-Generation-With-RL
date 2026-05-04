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
