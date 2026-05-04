import pytest

from src.rl.agent import DiffDockRLAgent
from src.rl.grpo import LinearSurrogateState
from src.rl.types import RLExample


def _example():
    return RLExample(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_input_path="ligand.sdf",
        predicted_pose_path="pose.sdf",
        ground_truth_pose_path="ligand_gt.sdf",
        sample_rank=2,
        sample_id=1,
        confidence_score=0.5,
    )


def test_agent_debug_linear_scores_and_checkpoint_roundtrip(tmp_path):
    agent = DiffDockRLAgent(
        surrogate_backend="debug_linear",
        linear_state=LinearSurrogateState(
            weights={"bias": 1.0, "confidence": 2.0, "inverse_rank": 4.0}
        ),
    )

    scores = agent.compute_surrogate_scores([_example()])
    checkpoint = tmp_path / "agent.json"
    agent.save_checkpoint(checkpoint, metadata={"step": 1})
    loaded = DiffDockRLAgent.load_checkpoint(checkpoint)

    assert scores == pytest.approx([4.0])
    assert loaded.compute_surrogate_scores([_example()]) == pytest.approx([4.0])


def test_agent_diffdock_loss_surrogate_raises_clear_error():
    agent = DiffDockRLAgent(surrogate_backend="diffdock_loss")

    with pytest.raises(NotImplementedError, match="DiffDock-loss surrogate"):
        agent.compute_surrogate_scores([_example()])


def test_agent_exact_logprobs_are_out_of_scope():
    agent = DiffDockRLAgent()

    with pytest.raises(NotImplementedError, match="Exact PPO"):
        agent.compute_exact_logprobs([])
