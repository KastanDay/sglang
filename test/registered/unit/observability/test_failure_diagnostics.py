import json
import os
from http import HTTPStatus
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq, msgpack_decode, msgpack_encode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.failure_diagnostics import (
    FRAME_MARKER,
    MAX_FRAME_BYTES,
    FailureDiagnosticWriter,
    classify_transfer_error,
    diagnostic_fields,
    normalize_cause,
    strip_diagnostic_fields,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Headers:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values.get(name)


def _request(**headers):
    return SimpleNamespace(headers=_Headers(headers))


def _finish_reason(cause="transfer timed out after 300000 ms"):
    return FINISH_ABORT(
        "client-safe error",
        HTTPStatus.INTERNAL_SERVER_ERROR,
        diagnostic_fields=diagnostic_fields(
            failure_stage="kv_transfer",
            error_kind="timeout",
            exception_class="KVTransferError",
            cause_detail=cause,
            transfer_id=1234,
            transfer_state="WaitingForInput",
            failure_timeout_ms=300000,
        ),
    ).to_json()


def _enabled_writer(monkeypatch, write_fd):
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: write_fd)
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(True):
        return FailureDiagnosticWriter("decode")


def test_diagnostic_fields_round_trip_through_msgpack():
    finish_reason = _finish_reason()
    decoded = msgpack_decode(
        msgpack_encode(AbortReq(rid="rid-1", finished_reason=finish_reason))
    )

    assert decoded.finished_reason == finish_reason


def test_writer_emits_bounded_frame_with_validated_correlation(monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        writer = _enabled_writer(monkeypatch, write_fd)
        result, event = writer.emit_finish_reason(
            finish_reason=_finish_reason("x" * 10000),
            rid="sglang-rid",
            request=_request(
                **{
                    "cf-workers-ai-diagnostic-account-label": "account_1",
                    "cf-workers-ai-diagnostic-parent-request-id": "parent-1",
                    "cf-workers-ai-diagnostic-batch-index": "4",
                    "x-request-id": "backend-4",
                }
            ),
            outcome="http_error",
        )
        frame = os.read(read_fd, MAX_FRAME_BYTES + 1)
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert result == "emitted"
    assert len(frame) <= MAX_FRAME_BYTES
    assert frame.startswith(b"\n" + FRAME_MARKER)
    payload = json.loads(frame.removeprefix(b"\n" + FRAME_MARKER))
    assert payload["truncated"] is True
    assert payload["account"] == "account_1"
    assert payload["parent_request_id"] == "parent-1"
    assert payload["request_id"] == "backend-4"
    assert payload["batch_index"] == 4
    assert event["cause_detail"] == payload["cause_detail"]


def test_writer_never_blocks_when_stdout_is_full(monkeypatch):
    writer = _enabled_writer(monkeypatch, 123)

    def would_block(_fd, _frame):
        raise BlockingIOError

    monkeypatch.setattr(os, "write", would_block)
    result, event = writer.emit_finish_reason(
        finish_reason=_finish_reason(),
        rid="rid",
        request=None,
        outcome="stream_abort",
    )

    assert result == "write_error"
    assert event is not None


def test_writer_opens_independent_nonblocking_stdout(monkeypatch):
    opened = {}

    def capture_open(path, flags):
        opened.update(path=path, flags=flags)
        raise OSError

    monkeypatch.setattr(os, "open", capture_open)
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(True):
        FailureDiagnosticWriter("decode")

    assert opened["path"] == "/proc/self/fd/1"
    assert opened["flags"] & os.O_NONBLOCK
    assert opened["flags"] & os.O_CLOEXEC
    assert opened["flags"] & os.O_APPEND


def test_disabled_writer_still_builds_failure_metric_event(monkeypatch):
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(False):
        writer = FailureDiagnosticWriter("prefill")

    result, event = writer.emit_finish_reason(
        finish_reason=_finish_reason(),
        rid="rid",
        request=None,
        outcome="http_error",
    )

    assert result == "disabled"
    assert event["account"] == "anonymous"
    assert event["leg"] == "prefill"


def test_unhandled_exception_does_not_copy_exception_message(monkeypatch):
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(False):
        writer = FailureDiagnosticWriter("decode")

    result, event = writer.emit_exception(
        exception=RuntimeError("private prompt text"),
        request=None,
    )

    assert result == "disabled"
    assert event["exception_class"] == "RuntimeError"
    assert "private prompt text" not in json.dumps(event)


def test_normalization_groups_variable_ids_and_numbers():
    first = normalize_cause(
        "KVTransferError room=123 request 2f1c5b9d-3eba-4c39-a815-3f953835f10d"
    )
    second = normalize_cause(
        "KVTransferError room=987 request 5bd16913-68d1-48c7-9dc2-74ba82b5dc78"
    )

    assert first == second


def test_transfer_timeouts_get_a_closed_error_kind():
    exception = RuntimeError("Request timed out in KVPoll.WaitingForInput")

    assert classify_transfer_error(exception, "transfer_failed") == "timeout"
    assert classify_transfer_error(
        RuntimeError("connection reset"), "transfer_failed"
    ) == ("transfer_failed")


def test_private_fields_are_removed_before_response():
    finish_reason = _finish_reason()
    strip_diagnostic_fields(finish_reason)

    assert set(finish_reason) == {"type", "message", "status_code", "err_type"}


def test_private_fields_are_removed_even_if_writer_fails():
    manager = object.__new__(TokenizerManager)

    class _FailingWriter:
        def emit_finish_reason(self, **_kwargs):
            raise RuntimeError("writer failed")

    manager.failure_diagnostic_writer = _FailingWriter()
    manager.enable_metrics = False
    finish_reason = _finish_reason()

    manager._consume_failure_diagnostic(
        finish_reason=finish_reason,
        obj=SimpleNamespace(rid="rid"),
        request=None,
        is_stream=False,
    )

    assert not any(key.startswith("_diag_") for key in finish_reason)
