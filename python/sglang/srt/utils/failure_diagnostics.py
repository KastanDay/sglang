# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

from sglang.srt.environ import envs

DIAGNOSTIC_PREFIX = "_diag_"
FRAME_MARKER = b"SGLANG_DIAGNOSTIC "
MAX_FRAME_BYTES = 2048

_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_BATCH_INDEX_RE = re.compile(r"^\d{1,9}$")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{12,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_NUMBER_RE = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")

_FAILURE_STAGES = {
    "decode",
    "grammar",
    "kv_bootstrap",
    "kv_transfer",
    "multimodal",
    "scheduler",
    "serving",
    "tokenizer",
    "unknown",
}
_ERROR_KINDS = {
    "anthropic_request_failed",
    "bootstrap_failed",
    "detokenization_failed",
    "encoder_transfer_failed",
    "handshake_failed",
    "invalid_token",
    "metadata_mismatch",
    "metadata_not_ready",
    "out_of_memory",
    "preempted",
    "priority_disabled",
    "queue_full",
    "retraction_host_exhausted",
    "reasoning_parse_failed",
    "running_timeout",
    "swa_reclaim_failed",
    "tokenization_failed",
    "token_count_failed",
    "timeout",
    "transfer_failed",
    "unhandled_exception",
    "unknown",
    "waiting_timeout",
}


def _validated_header(request: Any, name: str, pattern: re.Pattern) -> Optional[str]:
    if request is None:
        return None
    value = request.headers.get(name)
    if value is None or not pattern.fullmatch(value):
        return None
    return value


def normalize_cause(value: str) -> str:
    value = _UUID_RE.sub("<uuid>", value)
    value = _HEX_RE.sub("<hex>", value)
    value = _NUMBER_RE.sub("<n>", value)
    return _WHITESPACE_RE.sub(" ", value).strip().lower()


def classify_transfer_error(exception: Optional[Exception], default: str) -> str:
    if exception is None:
        return default
    try:
        detail = str(getattr(exception, "failure_reason", exception))[:512].lower()
    except Exception:
        return default
    return "timeout" if "timeout" in detail or "timed out" in detail else default


def _fingerprint(*parts: Any) -> str:
    joined = "\x1f".join(str(part or "") for part in parts)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def diagnostic_fields(
    *,
    failure_stage: Optional[str] = None,
    error_kind: Optional[str] = None,
    exception_class: Optional[str] = None,
    cause_detail: Optional[str] = None,
    transfer_id: Optional[Union[str, int]] = None,
    transfer_state: Optional[str] = None,
    failure_timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Return flat primitive fields safe for FinishReasonDict msgpack IPC."""
    if failure_stage is not None and failure_stage not in _FAILURE_STAGES:
        raise ValueError(f"Unsupported failure stage: {failure_stage}")
    if error_kind is not None and error_kind not in _ERROR_KINDS:
        raise ValueError(f"Unsupported error kind: {error_kind}")

    values = {
        "failure_stage": failure_stage,
        "error_kind": error_kind,
        "exception_class": exception_class,
        "cause_detail": cause_detail,
        "transfer_id": transfer_id,
        "transfer_state": transfer_state,
        "failure_timeout_ms": failure_timeout_ms,
    }
    return {
        f"{DIAGNOSTIC_PREFIX}{key}": value
        for key, value in values.items()
        if value is not None
    }


def strip_diagnostic_fields(finish_reason: Dict[str, Any]) -> None:
    for key in tuple(finish_reason):
        if key.startswith(DIAGNOSTIC_PREFIX):
            finish_reason.pop(key, None)


class FailureDiagnosticWriter:
    def __init__(self, role: str):
        self.enabled = envs.SGLANG_ENABLE_FAILURE_DIAGNOSTICS.get()
        self.role = role if role in ("decode", "prefill") else "unknown"
        self._fd: Optional[int] = None
        if self.enabled:
            try:
                # This file description has independent status flags, so
                # O_NONBLOCK does not alter normal stdout logging.
                self._fd = os.open(
                    "/proc/self/fd/1",
                    os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK | os.O_CLOEXEC,
                )
            except OSError:
                self._fd = None

    def emit_finish_reason(
        self,
        *,
        finish_reason: Dict[str, Any],
        rid: Optional[str],
        request: Any,
        outcome: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        try:
            return self._emit_finish_reason(
                finish_reason=finish_reason,
                rid=rid,
                request=request,
                outcome=outcome,
            )
        except Exception:
            # Diagnostics must never alter or delay the serving response.
            return "write_error", None

    def _emit_finish_reason(
        self,
        *,
        finish_reason: Dict[str, Any],
        rid: Optional[str],
        request: Any,
        outcome: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        try:
            status = int(finish_reason.get("status_code"))
        except (TypeError, ValueError):
            return "invalid_context", None
        if status not in (500, 503):
            return "invalid_context", None
        if outcome not in ("http_error", "stream_abort"):
            return "invalid_context", None

        account = (
            _validated_header(
                request,
                "cf-workers-ai-diagnostic-account-label",
                _ACCOUNT_RE,
            )
            or "anonymous"
        )
        parent_request_id = _validated_header(
            request,
            "cf-workers-ai-diagnostic-parent-request-id",
            _ID_RE,
        )
        request_id = _validated_header(request, "x-request-id", _ID_RE)
        batch_index_value = _validated_header(
            request,
            "cf-workers-ai-diagnostic-batch-index",
            _BATCH_INDEX_RE,
        )

        stage = str(finish_reason.get(f"{DIAGNOSTIC_PREFIX}failure_stage") or "unknown")
        kind = str(finish_reason.get(f"{DIAGNOSTIC_PREFIX}error_kind") or "unknown")
        if stage not in _FAILURE_STAGES or kind not in _ERROR_KINDS:
            return "invalid_context", None
        exception_class = str(
            finish_reason.get(f"{DIAGNOSTIC_PREFIX}exception_class") or ""
        )[:128]
        cause_detail = str(finish_reason.get(f"{DIAGNOSTIC_PREFIX}cause_detail") or "")
        transfer_state = finish_reason.get(f"{DIAGNOSTIC_PREFIX}transfer_state")
        transfer_id = finish_reason.get(f"{DIAGNOSTIC_PREFIX}transfer_id")
        timeout_ms = finish_reason.get(f"{DIAGNOSTIC_PREFIX}failure_timeout_ms")

        event: Dict[str, Any] = {
            "schema_version": 1,
            "event": "inference.failure",
            "producer_timestamp": datetime.now(timezone.utc).isoformat(),
            "abort_status": status,
            "outcome": outcome,
            "sglang_rid": str(rid)[:160] if rid is not None else None,
            "leg": self.role,
            "failure_stage": stage,
            "error_kind": kind,
            "exception_class": exception_class or None,
            "cause_detail": cause_detail,
            "transfer_id": str(transfer_id)[:160] if transfer_id is not None else None,
            "transfer_state": (
                str(transfer_state)[:64] if transfer_state is not None else None
            ),
            "failure_timeout_ms": int(timeout_ms) if timeout_ms is not None else None,
            "fingerprint": _fingerprint(self.role, stage, kind, exception_class),
            "cause_fingerprint": _fingerprint(
                self.role,
                stage,
                kind,
                exception_class,
                transfer_state,
                timeout_ms,
                normalize_cause(cause_detail),
            ),
            "severity": "ERROR",
            "truncated": False,
        }
        event["account"] = account
        if parent_request_id is not None:
            event["parent_request_id"] = parent_request_id
        if request_id is not None:
            event["request_id"] = request_id
        if batch_index_value is not None:
            event["batch_index"] = int(batch_index_value)

        event = {key: value for key, value in event.items() if value is not None}
        if not self.enabled:
            return "disabled", event

        frame = self._encode_bounded(event)
        if frame is None or self._fd is None:
            return "write_error", event

        try:
            written = os.write(self._fd, frame)
        except (BlockingIOError, OSError):
            return "write_error", event
        return ("emitted" if written == len(frame) else "write_error"), event

    def emit_exception(
        self,
        *,
        exception: Exception,
        request: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self.emit_finish_reason(
            finish_reason={
                "status_code": 500,
                **diagnostic_fields(
                    failure_stage="serving",
                    error_kind="unhandled_exception",
                    exception_class=type(exception).__name__,
                    cause_detail="Unhandled serving exception.",
                ),
            },
            rid=None,
            request=request,
            outcome="http_error",
        )

    def emit_serving_error(
        self,
        *,
        status_code: int,
        failure_stage: str,
        error_kind: str,
        exception: Exception,
        request: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self.emit_finish_reason(
            finish_reason={
                "status_code": status_code,
                **diagnostic_fields(
                    failure_stage=failure_stage,
                    error_kind=error_kind,
                    exception_class=type(exception).__name__,
                    cause_detail=f"Internal {failure_stage} failure.",
                ),
            },
            rid=None,
            request=request,
            outcome="http_error",
        )

    @staticmethod
    def _encode_bounded(event: Dict[str, Any]) -> Optional[bytes]:
        def encode() -> bytes:
            body = json.dumps(
                event,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            return b"\n" + FRAME_MARKER + body + b"\n"

        frame = encode()
        if len(frame) <= MAX_FRAME_BYTES:
            return frame

        event["truncated"] = True
        cause = str(event.get("cause_detail", ""))
        while cause and len(frame) > MAX_FRAME_BYTES:
            cause = cause[: max(0, len(cause) - (len(frame) - MAX_FRAME_BYTES))]
            event["cause_detail"] = cause
            frame = encode()
        if len(frame) <= MAX_FRAME_BYTES:
            return frame
        return None
