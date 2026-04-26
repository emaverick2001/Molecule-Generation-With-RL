import csv
import json

import pytest

from src.utils.artifact_logger import (
    save_csv,
    save_json,
    save_records_json,
    save_text,
)
from src.utils.schemas import GeneratedPose, MetricRecord


def test_save_json_writes_file(tmp_path):
    output_path = tmp_path / "metrics.json"

    save_json(
        {"top1_rmsd": 2.4, "success_at_5": 0.6},
        output_path,
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["top1_rmsd"] == 2.4
    assert data["success_at_5"] == 0.6


def test_save_json_creates_parent_directories(tmp_path):
    output_path = tmp_path / "artifacts" / "runs" / "metrics.json"

    save_json({"ok": True}, output_path)

    assert output_path.exists()


def test_save_records_json_writes_dataclass_records(tmp_path):
    output_path = tmp_path / "generated_samples_manifest.json"

    records = [
        GeneratedPose(
            complex_id="1abc",
            sample_id=0,
            pose_path="artifacts/generated_samples/1abc_sample_0.sdf",
            confidence_score=0.91,
        )
    ]

    save_records_json(records, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data[0]["complex_id"] == "1abc"
    assert data[0]["sample_id"] == 0
    assert data[0]["confidence_score"] == 0.91


def test_save_records_json_writes_dict_records(tmp_path):
    output_path = tmp_path / "metrics_records.json"

    records = [
        MetricRecord(
            complex_id="1abc",
            top1_rmsd=1.8,
            success_at_1=True,
            success_at_5=True,
            success_at_10=True,
        ).to_dict()
    ]

    save_records_json(records, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data[0]["complex_id"] == "1abc"
    assert data[0]["top1_rmsd"] == 1.8
    assert data[0]["success_at_1"] is True


def test_save_csv_writes_rows(tmp_path):
    output_path = tmp_path / "rewards.csv"

    rows = [
        {"complex_id": "1abc", "sample_id": 0, "reward": -1.8},
        {"complex_id": "1abc", "sample_id": 1, "reward": -2.3},
    ]

    save_csv(rows, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        loaded_rows = list(reader)

    assert loaded_rows[0]["complex_id"] == "1abc"
    assert loaded_rows[0]["sample_id"] == "0"
    assert loaded_rows[0]["reward"] == "-1.8"


def test_save_csv_raises_error_for_empty_rows(tmp_path):
    output_path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="Cannot save empty CSV"):
        save_csv([], output_path)


def test_save_text_writes_file(tmp_path):
    output_path = tmp_path / "errors.log"

    save_text("No errors found.", output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        text = f.read()

    assert text == "No errors found."