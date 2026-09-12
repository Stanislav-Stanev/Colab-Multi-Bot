"""Durable checkpointing that can never break `Run all` (plan §0, §5.2).

A SQLite checkpointer keeps interrupted cases alive across a kernel restart. The wheel may
be missing on any given runtime, so the import is guarded and the workflow degrades to
`MemorySaver` with one log line instead of failing the notebook.
"""
import sys

import pytest
from langgraph.checkpoint.memory import MemorySaver

import fraud_multi_agent as m


def test_checkpointer_falls_back_to_memory_when_sqlite_is_unavailable(monkeypatch, caplog):
    """A missing optional wheel must degrade, never raise (plan §0)."""
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)
    with caplog.at_level("INFO", logger=m.LOGGER_NAME):
        saver = m.make_checkpointer(":memory:")
    assert isinstance(saver, MemorySaver)
    assert "in-memory" in caplog.text.lower()


def test_a_present_but_incompatible_sqlite_wheel_also_falls_back(tmp_path, monkeypatch,
                                                                 caplog):
    """Importing is not the same as working.

    A `langgraph-checkpoint-sqlite` built against a different `langgraph-checkpoint`
    imports fine and then raises on the first checkpoint write — which is halfway through
    Run all, in every scenario cell. The real thing was observed as
    `AttributeError: 'JsonPlusSerializer' object has no attribute 'dumps'`, so the
    checkpointer is probed before it is trusted.
    """
    class BrokenSaver:
        def __init__(self, _conn):
            pass

        def put(self, *_args, **_kwargs):
            raise AttributeError("'JsonPlusSerializer' object has no attribute 'dumps'")

        def get(self, *_args, **_kwargs):
            return None

    fake_module = type(sys)("langgraph.checkpoint.sqlite")
    fake_module.SqliteSaver = BrokenSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", fake_module)

    with caplog.at_level("INFO", logger=m.LOGGER_NAME):
        saver = m.make_checkpointer(str(tmp_path / "broken.sqlite"))
    assert isinstance(saver, MemorySaver)
    assert "in-memory" in caplog.text.lower()


def test_sqlite_checkpointer_writes_a_usable_database(tmp_path):
    """The real thing: a case checkpointed through SQLite is readable back.

    This is the test that catches the connection-lifetime trap — `from_conn_string` hands
    back a context manager, and keeping only the saver it yields lets the connection be
    closed underneath the workflow, failing on the first checkpoint write in Colab.
    """
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    db = tmp_path / "cases.sqlite"
    saver = m.make_checkpointer(str(db))
    assert isinstance(saver, sqlite.SqliteSaver)

    config = {"configurable": {"thread_id": "sqlite-roundtrip", "checkpoint_ns": ""}}
    saver.put(config, {"v": 1, "id": "ckpt-1", "ts": "2026-01-01T00:00:00+00:00",
                       "channel_values": {"user_request": "review CUST-1001"},
                       "channel_versions": {}, "versions_seen": {}},
              {"source": "input", "step": 0, "parents": {}}, {})

    assert db.exists() and db.stat().st_size > 0
    restored = saver.get(config)
    assert restored["channel_values"]["user_request"] == "review CUST-1001"


def test_drive_is_off_by_default():
    """Mounting Drive opens an OAuth pop-up that would stall `Run all` (plan §0)."""
    assert m.USE_DRIVE is False


def test_checkpoint_path_is_local_unless_drive_is_requested(monkeypatch):
    assert "/content/drive" not in m.checkpoint_path(use_drive=False)
    assert m.checkpoint_path(use_drive=True).startswith("/content/drive")


def test_the_compiled_graph_has_a_checkpointer():
    assert m.graph.checkpointer is m.checkpointer
    assert m.checkpointer is not None


def test_a_paused_case_survives_recompiling_the_graph(monkeypatch):
    """What durability buys: the graph object can be rebuilt and the case resumes.

    This is the in-process half of the resume-after-restart demo — the same checkpointer
    backs a rebuilt graph, and the interrupted case is still there to resume.
    """
    from tests.test_graph_integration import FakeLLM, Script

    script = Script("CUST-1042", "BLOCK")
    monkeypatch.setattr(m, "get_llm", lambda: FakeLLM(script))
    thread = "durability-thread"

    started = m.start_workflow("Investigate CUST-1042.", thread_id=thread)
    assert started["status"] == "awaiting_human_review"

    rebuilt = m.build_graph(m.checkpointer)
    monkeypatch.setattr(m, "graph", rebuilt)
    done = m.resume_workflow(thread, {"type": "approve"})
    assert done["status"] == "completed"
    assert "BLOCKED" in done["final_output"]


# ---- side effects survive a replay without happening twice (review F6) -----------------


def test_the_same_notification_is_delivered_once_however_often_it_replays(tmp_path,
                                                                          monkeypatch):
    """LangGraph re-runs a node from its top when a case resumes mid-node.

    Without a key derived from durable state, a resumed case mints a second delivery id
    and the customer is told twice — in the mock it is invisible, in production it is the
    first incident.
    """
    import json
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    m._DELIVERED.clear()
    args = {"customer_id": "CUST-1042", "channel": "push", "subject": "Card blocked",
            "body": "We paused your card.", "case_id": "CASE-REPLAY",
            "idempotency_key": "CASE-REPLAY:notify:0"}

    first = json.loads(m.send_customer_notification.invoke(args))
    second = json.loads(m.send_customer_notification.invoke(args))

    assert first["status"] == "queued"
    assert second["delivery_id"] == first["delivery_id"], "a replay minted a second delivery"
    assert second["status"] == "duplicate"

    entries = [json.loads(line) for line
               in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    sent = [e for e in entries if e["event"] == "notification_sent"]
    assert len(sent) == 1, "one approval produced two deliveries in the audit trail"


def test_a_different_revision_is_a_different_message(tmp_path, monkeypatch):
    """Idempotency must not swallow a genuinely new message after a revision round."""
    import json
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    m._DELIVERED.clear()
    base = {"customer_id": "CUST-1042", "channel": "push", "subject": "s", "body": "b",
            "case_id": "CASE-REV"}

    first = json.loads(m.send_customer_notification.invoke(
        {**base, "idempotency_key": "CASE-REV:notify:0"}))
    second = json.loads(m.send_customer_notification.invoke(
        {**base, "idempotency_key": "CASE-REV:notify:1"}))

    assert first["delivery_id"] != second["delivery_id"]
    assert second["status"] == "queued"


def test_delivery_derives_its_key_from_durable_state(tmp_path, monkeypatch):
    """The key comes from the case and its revision — state that survives the restart."""
    import json
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    m._DELIVERED.clear()
    state = {"customer_id": "CUST-1042", "case_id": "CASE-KEY", "revision_count": 2,
             "recommended_action": "BLOCK", "human_decision": {"type": "approve"},
             "notification_draft": {"channel": "push", "language": "en",
                                    "subject": "s", "body": "b"},
             "final_output": "ACTION EXECUTED"}

    first = m.comms_deliver(dict(state))
    second = m.comms_deliver(dict(state))          # the node replays after a resume

    assert first["notification_result"]["status"] == "queued"
    assert second["notification_result"]["status"] == "duplicate"
    entries = [json.loads(line) for line
               in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([e for e in entries if e["event"] == "notification_sent"]) == 1
