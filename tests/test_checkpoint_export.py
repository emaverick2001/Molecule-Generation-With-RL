from pathlib import Path

import yaml

from src.rl.checkpoint_export import (
    export_native_grpo_checkpoint_for_diffdock_inference,
)


class FakeTorch:
    payload = None
    saved_payload = None
    saved_path = None
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

    @classmethod
    def save(cls, payload, path):
        cls.saved_payload = payload
        cls.saved_path = Path(path)
        cls.saved_path.write_text("checkpoint", encoding="utf-8")


def test_export_native_grpo_checkpoint_for_diffdock_inference(tmp_path):
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_text("placeholder", encoding="utf-8")
    source_model_dir = tmp_path / "score_model"
    source_model_dir.mkdir()
    (source_model_dir / "model_parameters.yml").write_text(
        yaml.safe_dump({"no_torsion": False}),
        encoding="utf-8",
    )
    output_model_dir = tmp_path / "exported_model"
    FakeTorch.payload = {
        "model_state_dict": {"layer.weight": 1.0},
        "optimizer_state_dict": {"ignored": True},
    }

    export = export_native_grpo_checkpoint_for_diffdock_inference(
        checkpoint_path=checkpoint,
        source_model_dir=source_model_dir,
        output_model_dir=output_model_dir,
        checkpoint_name="step_000.pt",
        torch_module=FakeTorch,
    )

    assert FakeTorch.loaded_path == checkpoint
    assert FakeTorch.loaded_map_location == "device:cpu"
    assert FakeTorch.saved_payload == {"layer.weight": 1.0}
    assert FakeTorch.saved_path == output_model_dir / "step_000.pt"
    assert (output_model_dir / "model_parameters.yml").is_file()
    assert export.checkpoint_name == "step_000.pt"
    assert export.checkpoint_path == str(output_model_dir / "step_000.pt")
