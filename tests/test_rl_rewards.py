import pytest

from src.rl.config import RewardConfig
from src.rl.rewards import (
    combine_rewards,
    compute_confidence_reward,
    compute_rmsd_reward,
    score_example,
)
from src.rl.types import RLExample, RewardComponent


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


def test_rmsd_reward_perfect_match_is_maximal(tmp_path):
    predicted = tmp_path / "predicted.sdf"
    reference = tmp_path / "reference.sdf"
    text = _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)])
    predicted.write_text(text, encoding="utf-8")
    reference.write_text(text, encoding="utf-8")

    reward = compute_rmsd_reward(predicted, reference)

    assert reward.valid is True
    assert reward.raw_value == pytest.approx(0.0)
    assert reward.value == pytest.approx(1.0)


def test_rmsd_reward_missing_gt_is_invalid(tmp_path):
    predicted = tmp_path / "predicted.sdf"
    predicted.write_text(_sdf([("C", 0.0, 0.0, 0.0)]), encoding="utf-8")

    reward = compute_rmsd_reward(predicted, None)

    assert reward.valid is False
    assert reward.value == -1.0
    assert "ground-truth" in reward.reason


def test_confidence_reward_accepts_logit():
    reward = compute_confidence_reward(0.0, mode="logit")

    assert reward.valid is True
    assert reward.value == pytest.approx(0.0)


def test_combine_rewards_renormalizes_missing_components():
    combined = combine_rewards(
        [
            RewardComponent("rmsd", 0.5, valid=True),
            RewardComponent("confidence", -1.0, valid=False, reason="missing"),
        ],
        weights={"rmsd": 1.0, "confidence": 1.0},
    )

    assert combined.valid is True
    assert combined.total == pytest.approx(0.5)


def test_score_example_uses_rmsd_and_confidence(tmp_path):
    predicted = tmp_path / "predicted.sdf"
    reference = tmp_path / "reference.sdf"
    text = _sdf([("C", 0.0, 0.0, 0.0)])
    predicted.write_text(text, encoding="utf-8")
    reference.write_text(text, encoding="utf-8")
    example = RLExample(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_input_path="ligand.sdf",
        predicted_pose_path=str(predicted),
        ground_truth_pose_path=str(reference),
        sample_rank=1,
        sample_id=0,
        confidence_score=0.0,
    )

    reward = score_example(
        example,
        RewardConfig(weights={"rmsd": 1.0, "confidence": 1.0}),
    )

    assert reward.valid is True
    assert reward.total == pytest.approx(0.5)
