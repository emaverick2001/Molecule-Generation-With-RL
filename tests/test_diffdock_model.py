from pathlib import Path

import pytest
import yaml

from src.rl.diffdock_model import (
    _extract_state_dict,
    load_diffdock_score_model,
    load_score_model_args,
)


class FakeInnerModel:
    def __init__(self):
        self.loaded_state_dict = None
        self.strict = None

    def load_state_dict(self, state_dict, *, strict):
        self.loaded_state_dict = state_dict
        self.strict = strict
        return "loaded"


class FakeModel:
    def __init__(self, *, wrapped=False):
        self.module = FakeInnerModel() if wrapped else None
        self.loaded_state_dict = None
        self.strict = None
        self.device = None
        self.eval_called = False

    def load_state_dict(self, state_dict, *, strict):
        self.loaded_state_dict = state_dict
        self.strict = strict
        return "loaded"

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeTorch:
    payload = None
    loaded_path = None
    loaded_map_location = None

    @staticmethod
    def device(name):
        return f"device:{name}"

    @classmethod
    def load(cls, path, *, map_location):
        cls.loaded_path = Path(path)
        cls.loaded_map_location = map_location
        return cls.payload


def test_load_score_model_args_reads_model_parameters(tmp_path):
    model_dir = tmp_path / "score_model"
    model_dir.mkdir()
    (model_dir / "model_parameters.yml").write_text(
        yaml.safe_dump({"receptor_radius": 30.0, "no_torsion": False}),
        encoding="utf-8",
    )

    args = load_score_model_args(model_dir=model_dir)

    assert args.receptor_radius == pytest.approx(30.0)
    assert args.no_torsion is False


def test_extract_state_dict_accepts_common_checkpoint_wrappers():
    assert _extract_state_dict({"state_dict": {"a": 1}}) == {"a": 1}
    assert _extract_state_dict({"model_state_dict": {"b": 2}}) == {"b": 2}
    assert _extract_state_dict({"model": {"c": 3}}) == {"c": 3}
    assert _extract_state_dict({"weight": 4}) == {"weight": 4}


def test_load_diffdock_score_model_loads_wrapped_module_state(tmp_path):
    model_dir = tmp_path / "score_model"
    model_dir.mkdir()
    checkpoint = model_dir / "best.pt"
    checkpoint.write_text("placeholder", encoding="utf-8")
    (model_dir / "model_parameters.yml").write_text(
        yaml.safe_dump({"no_torsion": False}),
        encoding="utf-8",
    )
    fake_model = FakeModel(wrapped=True)
    FakeTorch.payload = {"model_state_dict": {"weight": 1.0}}

    def fake_get_model(args, device, *, t_to_sigma, no_parallel, old):
        assert args.no_torsion is False
        assert device == "cuda"
        assert callable(t_to_sigma)
        assert no_parallel is False
        assert old is False
        return fake_model

    def fake_t_to_sigma(*args, **kwargs):
        return args, kwargs

    bundle = load_diffdock_score_model(
        repo_root=tmp_path,
        model_dir=model_dir,
        ckpt="best.pt",
        device="cuda",
        no_parallel=False,
        get_model_fn=fake_get_model,
        t_to_sigma_fn=fake_t_to_sigma,
        torch_module=FakeTorch,
    )

    assert FakeTorch.loaded_path == checkpoint
    assert FakeTorch.loaded_map_location == "device:cpu"
    assert fake_model.module.loaded_state_dict == {"weight": 1.0}
    assert fake_model.module.strict is True
    assert fake_model.device == "cuda"
    assert fake_model.eval_called is True
    assert bundle.checkpoint_path == checkpoint
