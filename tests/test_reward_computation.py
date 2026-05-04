from pathlib import Path

import pytest

from src.evaluation.metrics import (
    PoseMetricRecord,
    aggregate_topk_metrics,
    evaluate_generated_poses,
    load_pose_metrics_csv,
    save_pose_metrics_csv,
)
from src.pipeline.run_evaluation import build_negative_rmsd_reward_records
from src.utils.schemas import ComplexInput, GeneratedPose


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


def _metric(complex_id, sample_id, rmsd, valid=True):
    return PoseMetricRecord(
        complex_id=complex_id,
        sample_id=sample_id,
        rank=sample_id + 1,
        pose_path=f"{complex_id}_{sample_id}.sdf",
        reference_pose_path=f"{complex_id}_gt.sdf",
        rmsd=rmsd if valid else None,
        centroid_distance=rmsd if valid else None,
        rmsd_below_2=valid and rmsd < 2.0,
        rmsd_below_5=valid and rmsd < 5.0,
        valid=valid,
        error=None if valid else "bad pose",
    )


def test_evaluate_generated_poses_emits_valid_rows(tmp_path):
    reference_path = tmp_path / "ligand_gt.sdf"
    pose_path = tmp_path / "pose.sdf"
    reference_path.write_text(
        _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    pose_path.write_text(
        _sdf([("C", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    inputs = [
        ComplexInput(
            complex_id="1abc",
            protein_path=str(tmp_path / "protein.pdb"),
            ligand_path=str(tmp_path / "ligand.sdf"),
            ground_truth_pose_path=str(reference_path),
            split="smoke",
        )
    ]
    generated = [GeneratedPose("1abc", 0, str(pose_path))]

    records = evaluate_generated_poses(inputs, generated)

    assert len(records) == 1
    assert records[0].valid is True
    assert records[0].rmsd == pytest.approx(1.0)


def test_evaluate_generated_poses_emits_invalid_row_without_crashing(tmp_path):
    reference_path = tmp_path / "ligand_gt.sdf"
    reference_path.write_text(
        _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    inputs = [
        ComplexInput(
            complex_id="1abc",
            protein_path=str(tmp_path / "protein.pdb"),
            ligand_path=str(tmp_path / "ligand.sdf"),
            ground_truth_pose_path=str(reference_path),
            split="smoke",
        )
    ]
    generated = [GeneratedPose("1abc", 0, str(tmp_path / "missing_pose.sdf"))]

    records = evaluate_generated_poses(inputs, generated)

    assert records[0].valid is False
    assert "SDF file not found" in records[0].error


def test_aggregate_topk_metrics_computes_success_rates_and_best_of_n():
    records = [
        _metric("1abc", 0, 3.0),
        _metric("1abc", 1, 1.0),
        _metric("2xyz", 0, 1.5),
        _metric("2xyz", 1, 2.5),
        _metric("bad", 0, None, valid=False),
    ]

    aggregate = aggregate_topk_metrics(records, top_k=[1, 2], success_threshold=2.0)

    assert aggregate["num_complexes"] == 3
    assert aggregate["num_valid_poses"] == 4
    assert aggregate["num_invalid_poses"] == 1
    assert aggregate["success_at_1"] == 0.5
    assert aggregate["success_at_2"] == 1.0
    assert aggregate["best_of_n_mean_rmsd"] == pytest.approx(1.25)
    assert aggregate["median_best_rmsd"] == pytest.approx(1.25)
    assert aggregate["num_rmsd_gt_10"] == 0
    assert aggregate["fraction_rmsd_gt_10"] == 0.0


def test_aggregate_topk_metrics_reports_coverage_and_strict_success():
    records = [
        _metric("generated_success", 0, 1.0),
        _metric("generated_fail", 0, 3.0),
    ]

    aggregate = aggregate_topk_metrics(
        records,
        top_k=[1],
        success_threshold=2.0,
        attempted_complex_ids=[
            "generated_success",
            "generated_fail",
            "missing_generation",
        ],
    )

    assert aggregate["num_attempted_complexes"] == 3
    assert aggregate["num_generated_complexes"] == 2
    assert aggregate["generation_coverage"] == pytest.approx(2 / 3)
    assert aggregate["missing_generated_complexes"] == ["missing_generation"]
    assert aggregate["success_at_1"] == 0.5
    assert aggregate["strict_success_at_1"] == pytest.approx(1 / 3)


def test_aggregate_topk_metrics_counts_large_rmsd_outliers():
    records = [
        _metric("1abc", 0, 12.0),
        _metric("1abc", 1, 1.0),
        _metric("2xyz", 0, 15.0),
    ]

    aggregate = aggregate_topk_metrics(records, top_k=[1], success_threshold=2.0)

    assert aggregate["num_rmsd_gt_10"] == 2
    assert aggregate["fraction_rmsd_gt_10"] == pytest.approx(2 / 3)
    assert aggregate["median_best_rmsd"] == pytest.approx(8.0)


def test_aggregate_topk_metrics_returns_none_when_no_complex_has_enough_poses():
    aggregate = aggregate_topk_metrics([_metric("1abc", 0, 1.0)], top_k=[5])

    assert aggregate["success_at_5"] is None


def test_pose_metrics_csv_roundtrip(tmp_path):
    path = tmp_path / "pose_metrics.csv"
    records = [_metric("1abc", 0, 1.0)]

    save_pose_metrics_csv(records, path)
    loaded = load_pose_metrics_csv(path)

    assert loaded == records
    assert Path(path).is_file()


def test_build_negative_rmsd_reward_records_uses_real_pose_metrics():
    records = [
        _metric("1abc", 0, 3.25),
        _metric("2xyz", 0, None, valid=False),
    ]

    rewards = build_negative_rmsd_reward_records(records)

    assert len(rewards) == 1
    assert rewards[0].complex_id == "1abc"
    assert rewards[0].reward == -3.25
    assert rewards[0].reward_type == "negative_rmsd"
