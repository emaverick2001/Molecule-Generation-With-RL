import pytest

from src.rl.types import RLExample, RewardBreakdown, RewardComponent, RolloutRecord


def test_rl_example_roundtrip_dict():
    example = RLExample(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_input_path="ligand.sdf",
        predicted_pose_path="pose.sdf",
        ground_truth_pose_path="ligand_gt.sdf",
        sample_rank=1,
        sample_id=0,
        confidence_score=0.25,
        source_run_id="run1",
    )

    assert RLExample.from_dict(example.to_dict()) == example


def test_reward_breakdown_rejects_nan_total():
    with pytest.raises(ValueError, match="finite"):
        RewardBreakdown(total=float("nan"))


def test_rollout_record_roundtrip_dict():
    example = RLExample(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_input_path="ligand.sdf",
        predicted_pose_path="pose.sdf",
        ground_truth_pose_path="ligand_gt.sdf",
        sample_rank=1,
        sample_id=0,
    )
    reward = RewardBreakdown(
        total=1.0,
        components={"rmsd": RewardComponent("rmsd", 1.0, raw_value=0.0)},
    )
    record = RolloutRecord(
        group_id="1abc",
        example=example,
        reward=reward,
        advantage=0.0,
    )

    assert RolloutRecord.from_dict(record.to_dict()) == record
