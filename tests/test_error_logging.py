from src.utils.error_logger import append_error, append_exception


def test_append_error_writes_message(tmp_path):
    error_log_path = tmp_path / "errors.log"

    append_error(
        error_log_path,
        message="Generated pose file missing.",
        context={
            "complex_id": "1abc",
            "sample_id": 3,
        },
    )

    assert error_log_path.exists()

    text = error_log_path.read_text(encoding="utf-8")

    assert "ERROR" in text
    assert "Generated pose file missing." in text
    assert "complex_id: 1abc" in text
    assert "sample_id: 3" in text


def test_append_error_creates_parent_directories(tmp_path):
    error_log_path = tmp_path / "artifacts" / "runs" / "test_run" / "errors.log"

    append_error(
        error_log_path,
        message="Test error.",
    )

    assert error_log_path.exists()


def test_append_exception_writes_exception_and_traceback(tmp_path):
    error_log_path = tmp_path / "errors.log"

    try:
        raise ValueError("Invalid RMSD value")
    except Exception as exc:
        append_exception(
            error_log_path,
            exc,
            context={
                "stage": "evaluation",
                "complex_id": "1abc",
            },
        )

    text = error_log_path.read_text(encoding="utf-8")

    assert "EXCEPTION" in text
    assert "ValueError" in text
    assert "Invalid RMSD value" in text
    assert "stage: evaluation" in text
    assert "complex_id: 1abc" in text
    assert "traceback:" in text
    assert "Traceback" in text


def test_append_error_appends_multiple_entries(tmp_path):
    error_log_path = tmp_path / "errors.log"

    append_error(error_log_path, "First error.")
    append_error(error_log_path, "Second error.")

    text = error_log_path.read_text(encoding="utf-8")

    assert "First error." in text
    assert "Second error." in text
    assert text.count("ERROR") == 2