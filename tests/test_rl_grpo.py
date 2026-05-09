import pytest

from src.rl.grpo import (
    LinearSurrogateState,
    build_surrogate_score_rows,
    clipped_grpo_objective_term,
    compute_grpo_surrogate_loss,
    compute_surrogate_ratio,
    train_linear_grpo_step,
)
from src.rl.types import RLExample, RewardBreakdown, RolloutRecord


def _record(
    *,
    advantage: float,
    reward: float = 1.0,
    old_surrogate_score: float | None = 0.0,
    sample_id: int = 0,
) -> RolloutRecord:
    return RolloutRecord(
        group_id="1abc",
        example=RLExample(
            complex_id="1abc",
            protein_path="protein.pdb",
            ligand_input_path="ligand.sdf",
            predicted_pose_path=f"pose_{sample_id}.sdf",
            ground_truth_pose_path="ligand_gt.sdf",
            sample_rank=sample_id + 1,
            sample_id=sample_id,
            confidence_score=0.0,
        ),
        reward=RewardBreakdown(total=reward),
        advantage=advantage,
        old_surrogate_score=old_surrogate_score,
    )


def test_surrogate_ratio_uses_clipped_score_delta():
    ratio = compute_surrogate_ratio(100.0, 0.0, max_score_delta=2.0)

    assert ratio == pytest.approx(7.389056, rel=1e-5)


def test_clipped_grpo_objective_matches_option2_positive_advantage():
    ratio = compute_surrogate_ratio(2.0, 0.0, max_score_delta=20.0)
    term = clipped_grpo_objective_term(
        advantage=1.0,
        ratio=ratio,
        clip_epsilon=0.2,
    )

    assert term == pytest.approx(1.2)


def test_compute_grpo_surrogate_loss_uses_old_surrogate_scores():
    records = [
        _record(advantage=1.0, old_surrogate_score=0.0, sample_id=0),
        _record(advantage=-1.0, old_surrogate_score=0.0, sample_id=1),
    ]
    state = LinearSurrogateState(weights={"bias": 0.0, "confidence": 0.0, "inverse_rank": 0.0})

    assert compute_grpo_surrogate_loss(records, state) == pytest.approx(0.0)


def test_train_linear_grpo_step_reports_clipped_option2_metrics():
    records = [
        _record(advantage=1.0, old_surrogate_score=0.0, sample_id=0),
        _record(advantage=-1.0, old_surrogate_score=0.0, sample_id=1),
    ]
    state = LinearSurrogateState.initialized()

    updated_state, metrics = train_linear_grpo_step(
        records,
        state,
        learning_rate=0.1,
        clip_epsilon=0.2,
        max_score_delta=20.0,
    )
    rows = build_surrogate_score_rows(
        records,
        updated_state,
        clip_epsilon=0.2,
        max_score_delta=20.0,
    )

    assert metrics["clip_epsilon"] == pytest.approx(0.2)
    assert metrics["max_score_delta"] == pytest.approx(20.0)
    assert metrics["loss_after"] < metrics["loss_before"]
    assert rows[0]["old_surrogate_score"] == pytest.approx(0.0)
    assert rows[0]["surrogate_ratio"] > 1.0
    assert rows[0]["clipped_surrogate_ratio"] <= 1.2


def test_train_linear_grpo_step_stops_gradient_when_ratio_is_clipped():
    records = [_record(advantage=1.0, old_surrogate_score=0.0, sample_id=0)]
    saturated_state = LinearSurrogateState(
        weights={"bias": 2.0, "confidence": 0.0, "inverse_rank": 0.0}
    )

    updated_state, metrics = train_linear_grpo_step(
        records,
        saturated_state,
        learning_rate=0.1,
        clip_epsilon=0.2,
        max_score_delta=20.0,
    )

    assert metrics["gradients"]["bias"] == pytest.approx(0.0)
    assert updated_state.weights["bias"] == pytest.approx(2.0)
