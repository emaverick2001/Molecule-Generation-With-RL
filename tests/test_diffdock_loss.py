import pytest

from src.rl.diffdock_loss import (
    DiffDockLossBackend,
    DiffDockLossWeights,
    NativeDiffDockLossBackend,
    combine_diffdock_loss_components,
)
from src.rl.types import RLExample


def _example(sample_id=0):
    return RLExample(
        complex_id="1abc",
        protein_path="protein.pdb",
        ligand_input_path="ligand.sdf",
        predicted_pose_path="pose.sdf",
        ground_truth_pose_path="ligand_gt.sdf",
        sample_rank=sample_id + 1,
        sample_id=sample_id,
        confidence_score=0.0,
    )


def test_combine_diffdock_loss_components_from_mapping():
    components = combine_diffdock_loss_components(
        {
            "tr_loss": [1.0, 2.0],
            "rot_loss": [0.5, 0.25],
            "tor_loss": [0.1, 0.2],
        },
        weights=DiffDockLossWeights(tr=1.0, rot=2.0, tor=3.0),
    )

    assert components.total_loss == pytest.approx([2.3, 3.1])
    assert components.surrogate_scores == pytest.approx([-2.3, -3.1])


def test_combine_diffdock_loss_components_from_diffdock_tuple():
    components = combine_diffdock_loss_components(
        (
            [99.0, 99.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        )
    )

    assert components.total_loss == pytest.approx([9.0, 12.0])
    assert components.surrogate_scores == pytest.approx([-9.0, -12.0])


def test_diffdock_loss_backend_validates_score_count():
    backend = DiffDockLossBackend(
        lambda examples: {
            "tr_loss": [1.0],
            "rot_loss": [1.0],
            "tor_loss": [1.0],
        }
    )

    with pytest.raises(ValueError, match="returned 1 scores for 2 examples"):
        backend.score_examples([_example(0), _example(1)])


def test_diffdock_loss_backend_rows_match_examples():
    examples = [_example(0), _example(1)]
    backend = DiffDockLossBackend(
        lambda _: {
            "tr_loss": [1.0, 2.0],
            "rot_loss": [0.0, 0.0],
            "tor_loss": [0.0, 0.0],
        }
    )

    scores = backend.score_examples(examples)
    rows = backend.last_components.to_rows(examples)

    assert scores == pytest.approx([-1.0, -2.0])
    assert rows[0]["complex_id"] == "1abc"
    assert rows[1]["sample_id"] == 1
    assert rows[1]["diffdock_loss"] == pytest.approx(2.0)


def test_native_diffdock_loss_backend_accepts_current_diffdock_signature(tmp_path):
    class FakeModel:
        def __call__(self, batch):
            assert batch == ["batch"]
            return "tr_pred", "rot_pred", "tor_pred", "sidechain_pred"

    def fake_loss_function(
        tr_pred,
        rot_pred,
        tor_pred,
        sidechain_pred,
        *,
        data,
        t_to_sigma,
        device,
        tr_weight,
        rot_weight,
        tor_weight,
        backbone_weight,
        sidechain_weight,
        apply_mean,
        no_torsion,
    ):
        assert (tr_pred, rot_pred, tor_pred, sidechain_pred) == (
            "tr_pred",
            "rot_pred",
            "tor_pred",
            "sidechain_pred",
        )
        assert data == ["batch"]
        assert t_to_sigma is not None
        assert device == "cpu"
        assert (tr_weight, rot_weight, tor_weight) == (1.0, 1.0, 1.0)
        assert (backbone_weight, sidechain_weight) == (0.0, 0.0)
        assert apply_mean is False
        assert no_torsion is False
        return [0.0], [1.0], [2.0], [3.0], [0.0], [0.0]

    backend = NativeDiffDockLossBackend(
        repo_root=tmp_path,
        model=FakeModel(),
        t_to_sigma=lambda *args: args,
        batch_builder=lambda examples: ["batch"],
        device="cpu",
        loss_function=fake_loss_function,
    )

    assert backend.score_examples([_example()]) == pytest.approx([-6.0])
