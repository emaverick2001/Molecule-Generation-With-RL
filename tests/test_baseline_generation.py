import json
import sys
from pathlib import Path

import pytest

from src.generation.contract import validate_generated_pose_records
from src.generation.dry_run_generator import generate_dry_run_poses
from src.generation.generate_diffdock import (
    _collect_diffdock_output_poses,
    generate_diffdock_poses,
    preflight_diffdock_generation,
    run_diffdock_command,
)
from src.utils.artifact_logger import save_records_json
from src.utils.schemas import ComplexInput, GeneratedPose


def _records() -> list[ComplexInput]:
    return [
        ComplexInput(
            complex_id="1abc",
            protein_path="data/raw/pdbbind/1abc/protein.pdb",
            ligand_path="data/raw/pdbbind/1abc/ligand.sdf",
            ground_truth_pose_path="data/raw/pdbbind/1abc/ligand_gt.sdf",
            split="mini",
        ),
        ComplexInput(
            complex_id="2xyz",
            protein_path="data/raw/pdbbind/2xyz/protein.pdb",
            ligand_path="data/raw/pdbbind/2xyz/ligand.sdf",
            ground_truth_pose_path="data/raw/pdbbind/2xyz/ligand_gt.sdf",
            split="mini",
        ),
    ]


def test_dry_run_generator_returns_num_complexes_times_num_samples(tmp_path):
    records = _records()

    generated = generate_dry_run_poses(
        records=records,
        output_dir=tmp_path / "generated_samples",
        num_samples=3,
    )

    assert len(generated) == len(records) * 3


def test_dry_run_generator_records_have_valid_complex_ids(tmp_path):
    records = _records()
    valid_complex_ids = {record.complex_id for record in records}

    generated = generate_dry_run_poses(
        records=records,
        output_dir=tmp_path / "generated_samples",
        num_samples=2,
    )

    assert {pose.complex_id for pose in generated} == valid_complex_ids


def test_dry_run_generator_records_have_unique_complex_sample_keys(tmp_path):
    generated = generate_dry_run_poses(
        records=_records(),
        output_dir=tmp_path / "generated_samples",
        num_samples=2,
    )

    keys = {(pose.complex_id, pose.sample_id) for pose in generated}

    assert len(keys) == len(generated)


def test_dry_run_generator_pose_paths_are_inside_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "generated_samples"

    generated = generate_dry_run_poses(
        records=_records(),
        output_dir=output_dir,
        num_samples=2,
    )

    for pose in generated:
        pose_path = Path(pose.pose_path)
        assert pose_path.is_file()
        assert pose_path.resolve().is_relative_to(output_dir.resolve())


def test_generated_manifest_can_be_saved_and_reloaded(tmp_path):
    output_path = tmp_path / "generated_samples_manifest.json"
    generated = generate_dry_run_poses(
        records=_records(),
        output_dir=tmp_path / "generated_samples",
        num_samples=2,
    )

    save_records_json(generated, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    loaded_records = [GeneratedPose.from_dict(item) for item in loaded_data]

    assert loaded_records == generated


def test_validate_generated_pose_records_rejects_duplicate_keys(tmp_path):
    output_dir = tmp_path / "generated_samples"
    output_dir.mkdir()
    pose_path = output_dir / "1abc_sample_0.sdf"
    pose_path.write_text("pose\n", encoding="utf-8")
    duplicate = GeneratedPose("1abc", 0, str(pose_path))

    with pytest.raises(ValueError, match="Duplicate generated pose key"):
        validate_generated_pose_records(
            records=[_records()[0]],
            generated=[duplicate, duplicate],
            num_samples=2,
            output_dir=output_dir,
        )


def test_generate_diffdock_poses_runs_command_and_standardizes_outputs(tmp_path):
    records = _records()
    calls = []

    def fake_runner(command, cwd, stdout_path, stderr_path, timeout_seconds):
        calls.append((command, cwd, stdout_path, stderr_path, timeout_seconds))
        output_dir = Path(command[command.index("--out") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in range(2):
            (output_dir / f"rank{sample_id + 1}.sdf").write_text(
                f"pose {sample_id}\n",
                encoding="utf-8",
            )

    generated = generate_diffdock_poses(
        records=records,
        output_dir=tmp_path / "generated_samples",
        num_samples=2,
        command_template=[
            "diffdock",
            "--protein",
            "{protein_path}",
            "--ligand",
            "{ligand_path}",
            "--out",
            "{raw_output_dir}",
            "--samples",
            "{num_samples}",
        ],
        repo_dir=tmp_path,
        runner=fake_runner,
    )

    assert len(calls) == len(records)
    assert all(cwd == tmp_path.resolve() for _, cwd, _, _, _ in calls)
    assert [pose.pose_path for pose in generated] == [
        str(tmp_path / "generated_samples" / "1abc_sample_0.sdf"),
        str(tmp_path / "generated_samples" / "1abc_sample_1.sdf"),
        str(tmp_path / "generated_samples" / "2xyz_sample_0.sdf"),
        str(tmp_path / "generated_samples" / "2xyz_sample_1.sdf"),
    ]
    assert all(Path(pose.pose_path).is_file() for pose in generated)


def test_generate_diffdock_poses_raises_when_outputs_are_missing(tmp_path):
    def fake_runner(command, cwd, stdout_path, stderr_path, timeout_seconds):
        output_dir = Path(command[command.index("--out") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "only_pose.sdf").write_text("pose\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        generate_diffdock_poses(
            records=[_records()[0]],
            output_dir=tmp_path / "generated_samples",
            num_samples=2,
            command_template=["diffdock", "--out", "{output_dir}"],
            repo_dir=tmp_path,
            runner=fake_runner,
        )

    message = str(error.value)

    assert "expected at least 2" in message
    assert "Raw output directory contents" in message
    assert "only_pose.sdf" in message


def test_generate_diffdock_poses_can_skip_failed_complexes(tmp_path):
    records = _records()

    def fake_runner(command, cwd, stdout_path, stderr_path, timeout_seconds):
        output_dir = Path(command[command.index("--out") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_dir.name == "1abc":
            (output_dir / "rank1_confidence-0.42.sdf").write_text(
                "pose\n",
                encoding="utf-8",
            )
        else:
            (output_dir / "complex_0").mkdir()

    error_log_path = tmp_path / "errors.log"

    generated = generate_diffdock_poses(
        records=records,
        output_dir=tmp_path / "generated_samples",
        num_samples=1,
        command_template=["diffdock", "--out", "{raw_output_dir}"],
        repo_dir=tmp_path,
        runner=fake_runner,
        skip_failed_complexes=True,
        errors_log_path=error_log_path,
    )

    assert len(generated) == 1
    assert generated[0].complex_id == "1abc"
    assert generated[0].confidence_score == -0.42

    error_log = error_log_path.read_text(encoding="utf-8")

    assert "2xyz" in error_log
    assert "DiffDock produced 0 parseable ranked SDF files" in error_log


def test_generate_diffdock_poses_discovers_recursive_sdf_outputs(tmp_path):
    def fake_runner(command, cwd, stdout_path, stderr_path, timeout_seconds):
        output_dir = Path(command[command.index("--out") + 1])
        nested_output_dir = output_dir / "complex_outputs"
        nested_output_dir.mkdir(parents=True, exist_ok=True)
        (nested_output_dir / "rank1.sdf").write_text("pose\n", encoding="utf-8")

    generated = generate_diffdock_poses(
        records=[_records()[0]],
        output_dir=tmp_path / "generated_samples",
        num_samples=1,
        command_template=["diffdock", "--out", "{raw_output_dir}"],
        repo_dir=tmp_path,
        runner=fake_runner,
    )

    assert len(generated) == 1
    assert Path(generated[0].pose_path).is_file()


def test_collect_diffdock_output_poses_sorts_by_numeric_rank_and_confidence(tmp_path):
    output_dir = tmp_path / "raw_outputs"
    output_dir.mkdir()

    for filename in [
        "rank10_confidence-1.85.sdf",
        "rank2_confidence-0.75.sdf",
        "rank1.sdf",
        "rank1_confidence-0.70.sdf",
        "notes.sdf",
    ]:
        (output_dir / filename).write_text(filename, encoding="utf-8")

    poses = _collect_diffdock_output_poses(output_dir)

    assert [pose.rank for pose in poses] == [1, 2, 10]
    assert [pose.path.name for pose in poses] == [
        "rank1_confidence-0.70.sdf",
        "rank2_confidence-0.75.sdf",
        "rank10_confidence-1.85.sdf",
    ]
    assert [pose.confidence_score for pose in poses] == [-0.70, -0.75, -1.85]


def test_generate_diffdock_poses_uses_numeric_rank_order_and_confidence(tmp_path):
    def fake_runner(command, cwd, stdout_path, stderr_path, timeout_seconds):
        output_dir = Path(command[command.index("--out") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in [
            "rank10_confidence-1.85.sdf",
            "rank1_confidence-0.70.sdf",
            "rank2_confidence-0.75.sdf",
        ]:
            (output_dir / filename).write_text(filename, encoding="utf-8")

    generated = generate_diffdock_poses(
        records=[_records()[0]],
        output_dir=tmp_path / "generated_samples",
        num_samples=2,
        command_template=["diffdock", "--out", "{raw_output_dir}"],
        repo_dir=tmp_path,
        runner=fake_runner,
    )

    assert [pose.sample_id for pose in generated] == [0, 1]
    assert [pose.confidence_score for pose in generated] == [-0.70, -0.75]
    assert (tmp_path / "generated_samples" / "1abc_sample_0.sdf").read_text(
        encoding="utf-8"
    ) == "rank1_confidence-0.70.sdf"
    assert (tmp_path / "generated_samples" / "1abc_sample_1.sdf").read_text(
        encoding="utf-8"
    ) == "rank2_confidence-0.75.sdf"


def test_preflight_diffdock_generation_requires_repo_dir(tmp_path):
    config_path = tmp_path / "default_inference_args.yaml"
    config_path.write_text("inference\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="repo_dir"):
        preflight_diffdock_generation(
            records=_records(),
            repo_dir=tmp_path / "missing_diffdock",
            config_path=config_path,
            num_samples=1,
            command_template=["python", "-m", "inference"],
        )


def test_run_diffdock_command_failure_reports_log_paths_and_stderr_tail(tmp_path):
    stdout_path = tmp_path / "logs" / "failed.stdout.log"
    stderr_path = tmp_path / "logs" / "failed.stderr.log"

    with pytest.raises(RuntimeError) as error:
        run_diffdock_command(
            command=[
                sys.executable,
                "-c",
                "import sys; print('diffdock failed', file=sys.stderr); sys.exit(1)",
            ],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
        )

    message = str(error.value)

    assert "stdout log:" in message
    assert "stderr log:" in message
    assert "diffdock failed" in message
    assert stderr_path.read_text(encoding="utf-8").strip() == "diffdock failed"
