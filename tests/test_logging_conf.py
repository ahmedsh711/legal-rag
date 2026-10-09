import json

from legalrag.logging_conf import configure_logging, get_logger, request_id_var


def test_log_lines_are_json_with_request_id(capsys):
    configure_logging("INFO")
    token = request_id_var.set("req-123")
    try:
        get_logger("test").info("hello", stage="retrieve", duration_ms=12)
    finally:
        request_id_var.reset(token)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["request_id"] == "req-123"
    assert record["stage"] == "retrieve"
    assert record["level"] == "info"
    assert "timestamp" in record
