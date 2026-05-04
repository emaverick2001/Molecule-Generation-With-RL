from pathlib import Path

import pytest

from src.rl.config import RewardConfig, RolloutConfig
from src.rl.data import (
    export_complexes_to_diffdock_csv,
    group_examples_by_complex,
    join_samples_with_complex_manifest,
)
from src.rl.rollouts import (
    build_rollout_records,
    compute_group_advantages,
    compute_surrogate_ratios,
)
from src.utils.schemas import ComplexInput, GeneratedPose


def _complex(complex_id):
    return ComplexInput(
        complex_id=complex_id,
        protein_path=f"{complex_id}/protein.pdb",
        ligand_path=f"{complex_id}/ligand.sdf",
        ground_truth_pose_path=f"{complex_id}/ligand_gt.sdf",
        split="train",
    )


def test_join_samples_with_complex_manifest_success():
    examples = join_samples_with_complex_manifest(
        [GeneratedPose("1abc", 0, "pose.sdf", confidence_score=0.1)],
        [_complex("1abc")],
        source_run_id="run1",
    )

    assert examples[0].complex_id == "1abc"
    assert examples[0].sample_rank == 1
    assert examples[0].source_run_id == "run1"


def test_join_samples_with_complex_manifest_missing_id():
    with pytest.raises(ValueError, match="unknown complex_id"):
        join_samples_with_complex_manifest(
            [GeneratedPose("missing", 0, "pose.sdf")],
            [_complex("1abc")],
        )


def test_group_examples_by_complex_enforces_group_size():
    examples = join_samples_with_complex_manifest(
        [GeneratedPose("1abc", 0, "pose.sdf")],
        [_complex("1abc")],
    )

    with pytest.raises(ValueError, match="group size"):
        group_examples_by_complex(examples, expected_group_size=2)


def test_export_complexes_to_diffdock_csv_columns(tmp_path):
    path = export_complexes_to_diffdock_csv([_complex("1abc")], tmp_path / "in.csv")

    text = Path(path).read_text(encoding="utf-8")
    assert "complex_name,protein_path,ligand_description" in text
    assert "1abc,1abc/protein.pdb,1abc/ligand.sdf" in text


def test_group_advantages_zero_mean_per_group(tmp_path):
    text = """TestMol
  MVP

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
    reference = tmp_path / "ref.sdf"
    reference.write_text(text, encoding="utf-8")
    pose_a = tmp_path / "pose_a.sdf"
    pose_b = tmp_path / "pose_b.sdf"
    pose_a.write_text(text, encoding="utf-8")
    pose_b.write_text(text.replace("0.0000    0.0000", "1.0000    0.0000"), encoding="utf-8")

    complex_record = ComplexInput(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_path="ligand.sdf",
        ground_truth_pose_path=str(reference),
        split="train",
    )
    examples = join_samples_with_complex_manifest(
        [
            GeneratedPose("1abc", 0, str(pose_a)),
            GeneratedPose("1abc", 1, str(pose_b)),
        ],
        [complex_record],
    )
    records = build_rollout_records(
        examples,
        reward_cfg=RewardConfig(weights={"rmsd": 1.0}),
    )
    records = compute_group_advantages(
        records,
        rollout_cfg=RolloutConfig(min_valid_samples_per_complex=2),
    )

    assert sum(record.advantage for record in records) == pytest.approx(0.0)
    assert records[0].advantage > records[1].advantage


def test_surrogate_ratios_finite_after_clipping():
    ratios = compute_surrogate_ratios([100.0], [0.0], max_score_delta=2.0)

    assert ratios == pytest.approx([7.389056], rel=1e-5)
