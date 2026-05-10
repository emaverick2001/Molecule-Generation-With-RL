import pytest

from src.rl.grpo import (
    LinearSurrogateState,
    build_surrogate_score_rows,
    clipped_grpo_objective_term,
    compute_grpo_surrogate_loss,
    compute_surrogate_ratio,
    train_linear_grpo_step,
)
from src.rl.native_grpo import (
    native_loss_components_from_raw,
    run_native_diffdock_grpo_step_batched,
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


def test_native_loss_components_keep_differentiable_total_loss():
    torch = pytest.importorskip("torch")
    parameter = torch.tensor([2.0, 3.0], requires_grad=True)
    raw_total_loss = parameter * 2.0
    detached_component = raw_total_loss.detach()

    components = native_loss_components_from_raw(
        (
            raw_total_loss,
            detached_component + 1.0,
            detached_component + 2.0,
            detached_component + 3.0,
        ),
        torch_module=torch,
        device=torch.device("cpu"),
    )

    assert components["total_loss"].requires_grad
    assert components["scores"].requires_grad

    components["scores"].sum().backward()

    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([-2.0, -2.0])


def test_run_native_diffdock_grpo_step_batched_accumulates_gradients():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0.2]))

        def forward(self, batch):
            prediction = self.bias + batch.reshape(-1)
            zeros = torch.zeros_like(prediction)
            return prediction, zeros, zeros

    def tiny_loss_function(
        tr_pred,
        rot_pred,
        tor_pred,
        *,
        data,
        t_to_sigma,
        device,
        tr_weight=1.0,
        rot_weight=1.0,
        tor_weight=1.0,
        apply_mean=False,
        no_torsion=False,
    ):
        total = tr_pred.reshape(-1) ** 2
        zeros = torch.zeros_like(total)
        return total, total.detach(), zeros.detach(), zeros.detach()

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = float(model.bias.detach())

    result, state_dict = run_native_diffdock_grpo_step_batched(
        model=model,
        batches=[torch.tensor([1.0]), torch.tensor([-1.0])],
        advantages_by_batch=[[1.0], [-1.0]],
        loss_function=tiny_loss_function,
        t_to_sigma=lambda *args, **kwargs: None,
        device=torch.device("cpu"),
        optimizer=optimizer,
        torch_module=torch,
        max_grad_norm=1.0,
    )

    assert result.grad_norm is not None
    assert len(result.scores_before) == 2
    assert len(result.scores_after) == 2
    assert state_dict["bias"].shape == torch.Size([1])
    assert float(model.bias.detach()) != pytest.approx(before)
