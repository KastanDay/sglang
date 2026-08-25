import json
import os
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq, msgpack_decode, msgpack_encode
from sglang.srt.managers.rust_server import _client_safe_finish_reasons
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.failure_diagnostics import (
    FRAME_MARKER,
    MAX_FRAME_BYTES,
    FailureDiagnosticWriter,
    bounded_exception_cause,
    classify_transfer_error,
    client_safe_finish_reason,
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


def test_diagnostic_fields_never_break_serving_for_unknown_taxonomy():
    fields = diagnostic_fields(
        failure_stage="future_stage",
        error_kind="future_error",
    )

    assert fields["_diag_failure_stage"] == "unknown"
    assert fields["_diag_error_kind"] == "unknown"


def test_multimodal_processing_failure_is_supported():
    fields = diagnostic_fields(
        failure_stage="multimodal",
        error_kind="multimodal_processing_failed",
    )

    assert fields["_diag_error_kind"] == "multimodal_processing_failed"


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
        assert result == "emitted"
        frame = os.read(read_fd, MAX_FRAME_BYTES + 1)
    finally:
        os.close(read_fd)
        os.close(write_fd)

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


def test_writer_counts_corrupt_taxonomy_as_unknown(monkeypatch):
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(False):
        writer = FailureDiagnosticWriter("decode")

    finish_reason = _finish_reason()
    finish_reason["_diag_failure_stage"] = "future_stage"
    finish_reason["_diag_error_kind"] = "future_error"
    result, event = writer.emit_finish_reason(
        finish_reason=finish_reason,
        rid="rid",
        request=None,
        outcome="http_error",
    )

    assert result == "disabled"
    assert event["failure_stage"] == "unknown"
    assert event["error_kind"] == "unknown"


def test_writer_preserves_failure_metric_when_context_is_malformed(monkeypatch):
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(False):
        writer = FailureDiagnosticWriter("decode")

    finish_reason = _finish_reason()
    finish_reason["_diag_failure_timeout_ms"] = object()
    result, event = writer.emit_finish_reason(
        finish_reason=finish_reason,
        rid="rid",
        request=None,
        outcome="stream_abort",
    )

    assert result == "write_error"
    assert event == {
        "leg": "decode",
        "failure_stage": "unknown",
        "error_kind": "unknown",
        "abort_status": 500,
        "outcome": "stream_abort",
    }


def test_manager_counts_failure_when_context_is_malformed(monkeypatch):
    with envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.override(False):
        writer = FailureDiagnosticWriter("decode")
    observed = []
    manager = object.__new__(TokenizerManager)
    manager.failure_diagnostic_writer = writer
    manager.enable_metrics = True
    manager.metrics_collector = SimpleNamespace(
        observe_inference_failure=lambda event, result: observed.append((event, result))
    )
    finish_reason = _finish_reason()
    finish_reason["_diag_failure_timeout_ms"] = object()

    manager._consume_failure_diagnostic(
        finish_reason=finish_reason,
        obj=SimpleNamespace(rid="rid"),
        request=None,
        is_stream=True,
    )

    assert observed[0][0]["error_kind"] == "unknown"
    assert observed[0][1] == "write_error"


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


def test_client_safe_finish_reason_keeps_internal_source_for_diagnostics():
    finish_reason = _finish_reason("private prompt text")

    client_reason = client_safe_finish_reason(finish_reason)

    assert "private prompt text" not in json.dumps(client_reason)
    assert not any(key.startswith("_diag_") for key in client_reason)
    assert finish_reason["_diag_cause_detail"] == "private prompt text"


def test_known_prompt_echo_exception_is_replaced_with_constant():
    cause = bounded_exception_cause(ValueError("private prompt text"))

    assert cause == "Value validation failed."
    assert "private prompt text" not in cause


def test_unknown_exception_cause_is_normalized_and_bounded():
    cause = bounded_exception_cause(RuntimeError("Transfer 12345 failed  " + "X" * 700))

    assert cause.startswith("transfer <n> failed ")
    assert len(cause) == 512


def test_rust_egress_sanitizer_does_not_mutate_scheduler_diagnostics():
    finish_reason = _finish_reason("private prompt text")
    sanitized = _client_safe_finish_reasons([finish_reason, None])

    assert sanitized[1] is None
    assert "_diag_" not in json.dumps(sanitized)
    assert finish_reason["_diag_cause_detail"] == "private prompt text"


def test_private_fields_are_removed_even_if_writer_fails():
    manager = object.__new__(TokenizerManager)

    class _FailingWriter:
        def emit_finish_reason(self, **_kwargs):
            raise RuntimeError("writer failed")

    manager.failure_diagnostic_writer = _FailingWriter()
    manager.enable_metrics = False
    out = {"meta_info": {"finish_reason": _finish_reason()}}
    finish_reason = out["meta_info"]["finish_reason"]

    manager._consume_failure_diagnostic(
        finish_reason=finish_reason,
        obj=SimpleNamespace(rid="rid"),
        request=None,
        is_stream=False,
    )

    assert not any(key.startswith("_diag_") for key in finish_reason)
    assert "_diag_" not in json.dumps(out)


def test_anthropic_does_not_emit_a_diagnostic_twice():
    emitted = []
    serving = object.__new__(AnthropicServing)
    serving.openai_serving_chat = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            emit_serving_error=lambda **kwargs: emitted.append(kwargs)
        )
    )
    exception = RuntimeError("already reported")
    exception.sglang_failure_diagnostic_emitted = True

    serving._emit_serving_error(
        status_code=500,
        failure_stage="serving",
        error_kind="anthropic_request_failed",
        exception=exception,
        request=None,
    )

    assert emitted == []


def test_anthropic_diagnostic_is_optional_for_partial_test_managers():
    serving = object.__new__(AnthropicServing)
    serving.openai_serving_chat = SimpleNamespace(tokenizer_manager=object())

    serving._emit_serving_error(
        status_code=500,
        failure_stage="serving",
        error_kind="anthropic_request_failed",
        exception=RuntimeError("failure"),
        request=None,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
