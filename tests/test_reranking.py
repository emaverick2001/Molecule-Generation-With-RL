import pytest

from src.evaluation.metrics import evaluate_generated_poses
from src.evaluation.reranking import (
    RerankedPose,
    load_reranked_manifest,
    rerank_generated_poses,
    save_reranked_manifest,
    summarize_reranking,
)
from src.evaluation.reranking_comparison import (
    compare_reranking_strategies,
    summarize_reranking_comparison,
)
from src.rewards.confidence_reward import (
    build_confidence_reward_records,
    require_confidence_scores,
    transform_confidence_score,
)
from src.utils.schemas import ComplexInput, GeneratedPose, RewardRecord


def _poses() -> list[GeneratedPose]:
    return [
        GeneratedPose("1abc", 0, "sample_0.sdf", confidence_score=0.2),
        GeneratedPose("1abc", 1, "sample_1.sdf", confidence_score=0.9),
        GeneratedPose("1abc", 2, "sample_2.sdf", confidence_score=0.9),
        GeneratedPose("2xyz", 0, "other_0.sdf", confidence_score=0.5),
    ]


def _rewards() -> list[RewardRecord]:
    return [
        RewardRecord("1abc", 0, 0.2, "confidence", True),
        RewardRecord("1abc", 1, 0.9, "confidence", True),
        RewardRecord("1abc", 2, 0.9, "confidence", True),
        RewardRecord("2xyz", 0, 0.5, "confidence", True),
    ]


def test_reranking_sorts_descending_reward_within_each_complex():
    reranked = rerank_generated_poses(_poses(), _rewards())

    one_abc = [record for record in reranked if record.complex_id == "1abc"]

    assert [record.sample_id for record in one_abc] == [1, 2, 0]
    assert [record.reranked_rank for record in one_abc] == [1, 2, 3]


def test_reranking_ties_break_by_sample_id():
    reranked = rerank_generated_poses(_poses(), _rewards(), tie_breaker="sample_id")
    one_abc = [record for record in reranked if record.complex_id == "1abc"]

    assert [record.sample_id for record in one_abc[:2]] == [1, 2]


def test_missing_reward_row_raises_clear_error():
    with pytest.raises(ValueError, match="Missing reward row"):
        rerank_generated_poses(_poses(), _rewards()[:-1])


def test_reranked_manifest_roundtrip(tmp_path):
    path = tmp_path / "reranked_generated_samples_manifest.json"
    reranked = rerank_generated_poses(_poses(), _rewards())

    save_reranked_manifest(reranked, path)
    loaded = load_reranked_manifest(path)

    assert loaded == reranked


def test_reranking_does_not_duplicate_or_rename_pose_files():
    original_pose_paths = {pose.pose_path for pose in _poses()}
    reranked = rerank_generated_poses(_poses(), _rewards())

    assert {record.pose_path for record in reranked} == original_pose_paths


def test_confidence_only_reranking_is_idempotent_when_already_sorted():
    poses = [
        GeneratedPose("1abc", 0, "sample_0.sdf", confidence_score=0.9),
        GeneratedPose("1abc", 1, "sample_1.sdf", confidence_score=0.8),
    ]
    rewards = build_confidence_reward_records(poses)
    reranked = rerank_generated_poses(poses, rewards)

    assert [record.sample_id for record in reranked] == [0, 1]
    assert summarize_reranking(reranked)["num_rank_changed"] == 0


def test_require_confidence_scores_raises_when_missing():
    with pytest.raises(ValueError, match="Missing confidence_score"):
        require_confidence_scores([GeneratedPose("1abc", 0, "pose.sdf")])


def test_transform_confidence_score_supports_sigmoid():
    assert transform_confidence_score(0.0, transform="sigmoid") == pytest.approx(0.5)


def test_evaluation_uses_manifest_order_for_rank(tmp_path):
    reference_path = tmp_path / "ligand_gt.sdf"
    reference_path.write_text(
        """TestMol
  MVP

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
""",
        encoding="utf-8",
    )
    first_pose = tmp_path / "sample_5.sdf"
    second_pose = tmp_path / "sample_1.sdf"
    first_pose.write_text(reference_path.read_text(encoding="utf-8"), encoding="utf-8")
    second_pose.write_text(reference_path.read_text(encoding="utf-8"), encoding="utf-8")

    input_records = [
        ComplexInput(
            complex_id="1abc",
            protein_path="protein.pdb",
            ligand_path="ligand.sdf",
            ground_truth_pose_path=str(reference_path),
            split="smoke",
        )
    ]
    generated_records = [
        RerankedPose("1abc", 5, str(first_pose), 6, 1, 0.9, "confidence"),
        RerankedPose("1abc", 1, str(second_pose), 2, 2, 0.8, "confidence"),
    ]

    metric_records = evaluate_generated_poses(input_records, generated_records)

    assert [(record.sample_id, record.rank) for record in metric_records] == [
        (5, 1),
        (1, 2),
    ]


def test_reranking_comparison_reports_original_confidence_and_oracle():
    metric_records = [
        _metric_record("1abc", sample_id=0, rank=1, rmsd=5.0),
        _metric_record("1abc", sample_id=1, rank=2, rmsd=1.0),
        _metric_record("1abc", sample_id=2, rank=3, rmsd=2.0),
        _metric_record("2xyz", sample_id=0, rank=1, rmsd=1.5),
        _metric_record("2xyz", sample_id=1, rank=2, rmsd=3.0),
    ]
    generated_records = [
        GeneratedPose("1abc", 0, "a0.sdf", confidence_score=0.9),
        GeneratedPose("1abc", 1, "a1.sdf", confidence_score=0.2),
        GeneratedPose("1abc", 2, "a2.sdf", confidence_score=0.1),
        GeneratedPose("2xyz", 0, "b0.sdf", confidence_score=0.7),
        GeneratedPose("2xyz", 1, "b1.sdf", confidence_score=0.1),
    ]

    comparison = compare_reranking_strategies(metric_records, generated_records)
    summary = summarize_reranking_comparison(comparison)
    first = next(record for record in comparison if record.complex_id == "1abc")

    assert first.original_top_sample_id == 0
    assert first.confidence_top_sample_id == 0
    assert first.oracle_sample_id == 1
    assert first.confidence_selected_oracle is False
    assert summary["original_success_at_1"] == 0.5
    assert summary["confidence_success_at_1"] == 0.5
    assert summary["oracle_success_at_1"] == 1.0


def _metric_record(complex_id, sample_id, rank, rmsd):
    from src.evaluation.metrics import PoseMetricRecord

    return PoseMetricRecord(
        complex_id=complex_id,
        sample_id=sample_id,
        rank=rank,
        pose_path=f"{complex_id}_{sample_id}.sdf",
        reference_pose_path=f"{complex_id}_gt.sdf",
        rmsd=rmsd,
        centroid_distance=rmsd,
        rmsd_below_2=rmsd < 2.0,
        rmsd_below_5=rmsd < 5.0,
        valid=True,
    )
