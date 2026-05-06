import pytest

from src.rl.diffdock_batch_builder import (
    DiffDockBatchBuilderConfig,
    DiffDockGeneratedPoseBatchBuilder,
    config_from_score_model_args,
)
from src.rl.types import RLExample


class FakeTensor:
    def __init__(self, data):
        self.data = [[float(value) for value in row] for row in data]
        self.shape = (len(self.data), len(self.data[0]) if self.data else 0)

    def __sub__(self, other):
        if not isinstance(other, FakeTensor):
            return NotImplemented

        if other.shape[0] == 1:
            other_data = other.data * self.shape[0]
        else:
            other_data = other.data

        return FakeTensor(
            [
                [left_value - right_value for left_value, right_value in zip(left, right)]
                for left, right in zip(self.data, other_data)
            ]
        )


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(data, *, dtype):
        assert dtype == FakeTorch.float32
        return FakeTensor(data)


class FakeLigand:
    def __init__(self, atom_count):
        self.pos = FakeTensor([[0.0, 0.0, 0.0] for _ in range(atom_count)])


class FakeGraph:
    def __init__(self, name, *, atom_count=2, success=True):
        self.name = name
        self.values = {
            "ligand": FakeLigand(atom_count),
            "success": success,
        }
        self.original_center = FakeTensor([[1.0, 2.0, 3.0]])
        self.noise_applied = False
        self.device = None

    def __contains__(self, key):
        return key in self.values

    def __getitem__(self, key):
        return self.values[key]

    def to(self, device):
        self.device = device
        return self


class FakeInferenceDataset:
    last_kwargs = None
    atom_count = 2
    success = True

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.graphs = [
            FakeGraph(
                name,
                atom_count=type(self).atom_count,
                success=type(self).success,
            )
            for name in kwargs["complex_names"]
        ]

    def __getitem__(self, index):
        return self.graphs[index]


class FakeNoiseTransform:
    last_args = None
    last_kwargs = None

    def __init__(self, *args, **kwargs):
        type(self).last_args = args
        type(self).last_kwargs = kwargs

    def __call__(self, graph):
        graph.noise_applied = True
        return graph


class FakeDataLoader:
    def __init__(self, *, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        yield self.dataset


class FakeConformer:
    def __init__(self, positions):
        self.positions = positions

    def GetPositions(self):
        return self.positions


class FakeMol:
    def __init__(self, positions):
        self.positions = positions

    def GetConformer(self):
        return FakeConformer(self.positions)


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type


def _example(sample_id=0):
    return RLExample(
        complex_id="1abc",
        protein_path=f"protein_{sample_id}.pdb",
        ligand_input_path=f"ligand_{sample_id}.sdf",
        predicted_pose_path=f"pose_{sample_id}.sdf",
        ground_truth_pose_path=f"ligand_gt_{sample_id}.sdf",
        sample_rank=sample_id + 1,
        sample_id=sample_id,
        confidence_score=0.0,
    )


def _read_molecule(path, *, remove_hs, sanitize):
    assert remove_hs is False
    assert sanitize is True
    return FakeMol(
        [
            [10.0 + len(path), 20.0, 30.0],
            [11.0 + len(path), 21.0, 31.0],
        ]
    )


def _remove_hs(mol):
    return mol


def _builder(device="cpu", *, dataset_cls=FakeInferenceDataset):
    return DiffDockGeneratedPoseBatchBuilder(
        config=DiffDockBatchBuilderConfig(
            repo_root="unused",
            work_dir="unused",
            lm_embeddings=False,
            receptor_radius=12.0,
            c_alpha_max_neighbors=5,
            remove_hs=False,
            all_atoms=True,
            atom_radius=4.0,
            atom_max_neighbors=6,
            knn_only_graph=True,
            no_torsion=True,
            noise_alpha=0.7,
            noise_beta=0.8,
            minimum_t=0.05,
            crop_beyond_cutoff=20.0,
            include_miscellaneous_atoms=True,
        ),
        t_to_sigma=lambda *args: args,
        device=device,
        inference_dataset_cls=dataset_cls,
        noise_transform_cls=FakeNoiseTransform,
        data_loader_cls=FakeDataLoader,
        read_molecule_fn=_read_molecule,
        remove_hs_fn=_remove_hs,
        tensor_fn=FakeTorch,
    )


def test_config_from_score_model_args_uses_diffdock_defaults():
    class Args:
        receptor_radius = 14.0
        c_alpha_max_neighbors = 7
        remove_hs = True
        all_atoms = True
        atom_radius = 3.5
        atom_max_neighbors = 9
        not_knn_only_graph = False
        no_torsion = True
        crop_beyond = 18.0

    config = config_from_score_model_args(
        Args(),
        repo_root="repo",
        work_dir="work",
        lm_embeddings=False,
    )

    assert config.repo_root == "repo"
    assert config.work_dir == "work"
    assert config.lm_embeddings is False
    assert config.receptor_radius == pytest.approx(14.0)
    assert config.c_alpha_max_neighbors == 7
    assert config.remove_hs is True
    assert config.all_atoms is True
    assert config.atom_radius == pytest.approx(3.5)
    assert config.atom_max_neighbors == 9
    assert config.knn_only_graph is True
    assert config.no_torsion is True
    assert config.crop_beyond_cutoff == pytest.approx(18.0)


def test_batch_builder_reuses_inference_graph_and_replaces_pose_coordinates():
    examples = [_example(0), _example(1)]
    batch = _builder()(examples)

    assert len(batch) == 2
    assert FakeInferenceDataset.last_kwargs["complex_names"] == [
        "1abc_sample_0",
        "1abc_sample_1",
    ]
    assert FakeInferenceDataset.last_kwargs["protein_files"] == [
        "protein_0.pdb",
        "protein_1.pdb",
    ]
    assert FakeInferenceDataset.last_kwargs["ligand_descriptions"] == [
        "pose_0.sdf",
        "pose_1.sdf",
    ]
    assert FakeInferenceDataset.last_kwargs["lm_embeddings"] is False
    assert FakeInferenceDataset.last_kwargs["receptor_radius"] == pytest.approx(12.0)
    assert FakeInferenceDataset.last_kwargs["all_atoms"] is True
    assert FakeInferenceDataset.last_kwargs["knn_only_graph"] is True

    assert FakeNoiseTransform.last_kwargs == {
        "alpha": 0.7,
        "beta": 0.8,
        "include_miscellaneous_atoms": True,
        "crop_beyond_cutoff": 20.0,
        "minimum_t": 0.05,
    }
    assert FakeNoiseTransform.last_args[1:] == (True, True)

    first_pose = batch[0]["ligand"].pos
    assert first_pose.data == [
        [19.0, 18.0, 27.0],
        [20.0, 19.0, 28.0],
    ]
    assert all(graph.noise_applied for graph in batch)


def test_batch_builder_returns_cuda_graph_list():
    examples = [_example(0), _example(1)]
    device = FakeDevice("cuda")
    batch = _builder(device=device)(examples)

    assert isinstance(batch, list)
    assert [graph.device for graph in batch] == [device, device]


def test_batch_builder_rejects_failed_diffdock_graph():
    class FailedDataset(FakeInferenceDataset):
        success = False

    with pytest.raises(ValueError, match="graph construction failed"):
        _builder(dataset_cls=FailedDataset)([_example()])


def test_batch_builder_rejects_atom_count_mismatch():
    class OneAtomDataset(FakeInferenceDataset):
        atom_count = 1

    with pytest.raises(ValueError, match="atom count does not match"):
        _builder(dataset_cls=OneAtomDataset)([_example()])
