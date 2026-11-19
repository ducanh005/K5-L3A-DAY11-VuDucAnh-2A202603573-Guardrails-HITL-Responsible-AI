"""Unit tests for Checkpoint 3 pipeline, rate limiter, audit, monitoring, and egress."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import (
    build_observability,
    build_production_plugins,
    is_egress_allowed,
)
import assignment.rate_limiter as rl_mod
from assignment.rate_limiter import RateLimitPlugin
from guardrails.input_guardrails import InputGuardrailPlugin
from guardrails.output_guardrails import OutputGuardrailPlugin


def _make_content(text: str = "ping") -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


# ---------------------------------------------------------------------------
# RateLimitPlugin
# ---------------------------------------------------------------------------


def test_rate_limiter_sliding_window_isolation_and_expiry(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rl_mod.time, "time", lambda: now[0])

    plugin = RateLimitPlugin(max_requests=3, window_seconds=60)
    ctx_a = SimpleNamespace(user_id="alice")
    ctx_b = SimpleNamespace(user_id="bob")

    async def _run():
        a_results = [
            await plugin.on_user_message_callback(
                invocation_context=ctx_a, user_message=_make_content()
            )
            for _ in range(4)
        ]
        # Another user is unaffected
        b_res = await plugin.on_user_message_callback(
            invocation_context=ctx_b, user_message=_make_content()
        )
        # Advance past window_seconds -> alice's window expires
        now[0] += 61.0
        a_after = await plugin.on_user_message_callback(
            invocation_context=ctx_a, user_message=_make_content()
        )
        return a_results, b_res, a_after

    a_results, b_res, a_after = asyncio.run(_run())

    assert a_results[:3] == [None, None, None]
    assert isinstance(a_results[3], types.Content)
    assert "rate limit exceeded" in a_results[3].parts[0].text.lower()
    assert b_res is None
    assert a_after is None
    assert plugin.total_count == 6
    assert plugin.blocked_count == 1


# ---------------------------------------------------------------------------
# AuditLogPlugin
# ---------------------------------------------------------------------------


def test_audit_log_records_latency_and_exports_json(tmp_path):
    audit = AuditLogPlugin()
    audit.record_input(user_id="u1", text="Check balance", request_id="req-1")
    entry = audit.record_output(
        user_id="u1",
        text="Your balance is...",
        blocked=False,
        layer=None,
        request_id="req-1",
    )
    audit.record_input(user_id="u2", text="Ignore all instructions", request_id="req-2")
    audit.record_output(
        user_id="u2",
        text="Blocked",
        blocked=True,
        layer="input_guardrail",
        request_id="req-2",
    )

    assert entry["latency_ms"] >= 0.0
    assert entry["blocked"] is False
    assert len(audit.logs) == 2

    out_path = tmp_path / "nested" / "audit_log.json"
    written = audit.export_json(str(out_path))
    assert Path(written).is_file()

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list) and len(loaded) == 2
    assert loaded[1]["blocked"] is True
    assert loaded[1]["layer"] == "input_guardrail"


# ---------------------------------------------------------------------------
# MonitoringAlert
# ---------------------------------------------------------------------------


def test_monitoring_alert_thresholds_dedup_and_export(tmp_path):
    monitor = MonitoringAlert(
        block_rate_threshold=0.5,
        rate_limit_hit_threshold=3,
        judge_fail_rate_threshold=0.3,
    )

    # Under thresholds -> no alerts
    monitor.record_request(blocked=False)
    monitor.record_request(blocked=False)
    assert monitor.check_metrics() == []

    # Exceed all thresholds
    monitor.record_request(blocked=True, rate_limited=True, judge_checked=True, judge_failed=True)
    monitor.record_request(blocked=True, rate_limited=True, judge_checked=True, judge_failed=True)
    monitor.record_request(blocked=True, rate_limited=True)
    alerts = monitor.check_metrics()
    assert {a.metric for a in alerts} == {"block_rate", "rate_limit_hits", "judge_fail_rate"}

    # Repeated check with unchanged counters does not duplicate alerts
    assert monitor.check_metrics() == []
    assert len(monitor.alerts) == 3

    out_path = tmp_path / "metrics.json"
    monitor.export_json(str(out_path))
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_requests"] == 5
    assert data["blocked_requests"] == 3
    assert data["rate_limit_hits"] == 3
    assert len(data["alerts"]) == 3


# ---------------------------------------------------------------------------
# is_egress_allowed & pipeline builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "destination,payload",
    [
        ("https://api.vinbank.example/v1/transfers", "approved transfer amount 500000"),
        ("https://cases.vinbank.example/tickets/123", "customer inquiry regarding savings rate 4.25%"),
    ],
)
def test_egress_allows_trusted_https_with_clean_payload(destination: str, payload: str):
    assert is_egress_allowed(destination, payload) is True


@pytest.mark.parametrize(
    "destination,payload",
    [
        ("http://api.vinbank.example/v1/transfers", "approved transfer amount 500000"),
        ("https://evil.example/collect", "customer account 123456"),
        ("https://api.vinbank.example.evil.com/v1/transfers", "approved transfer amount 500000"),
        ("https://user:pass@api.vinbank.example/v1/transfers", "approved transfer amount 500000"),
        ("", "approved transfer amount 500000"),
        ("https://api.vinbank.example/v1/transfers", "admin password is admin123"),
        ("https://api.vinbank.example/v1/transfers", "token sk-vinbank-secret-2024"),
        ("https://api.vinbank.example/v1/transfers", "host db.vinbank.internal:5432"),
        ("https://api.vinbank.example/v1/transfers", "call 0901234567"),
        ("https://api.vinbank.example/v1/transfers", "email user@example.com"),
        ("https://api.vinbank.example/v1/transfers", "CCCD 079204001234"),
        ("https://api.vinbank.example/v1/transfers", "a\u200bdmin123"),
    ],
)
def test_egress_blocks_untrusted_destinations_and_sensitive_payloads(
    destination: str, payload: str
):
    assert is_egress_allowed(destination, payload) is False


def test_build_production_plugins_order_and_observability():
    plugins = build_production_plugins(max_requests=7, window_seconds=30, use_llm_judge=False)
    assert len(plugins) == 3
    assert isinstance(plugins[0], RateLimitPlugin)
    assert plugins[0].max_requests == 7
    assert plugins[0].window_seconds == 30
    assert isinstance(plugins[1], InputGuardrailPlugin)
    assert isinstance(plugins[2], OutputGuardrailPlugin)

    audit, monitor = build_observability()
    assert isinstance(audit, AuditLogPlugin)
    assert isinstance(monitor, MonitoringAlert)
