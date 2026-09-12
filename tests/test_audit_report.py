"""The final section: the run's audit trail, rendered for a human to read.

Section 8.3 shows the last twelve entries and proves the chain. This is the other
question a reviewer asks — "show me everything that happened, case by case" — which a
flat tail of a JSONL file does not answer.
"""
import json

import fraud_multi_agent as m


def _write(path, entries):
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
                    encoding="utf-8")


def _entry(event, run_id="run-a", case_id="CASE-1", customer_id="CUST-1042", **fields):
    return {"ts": "2026-09-12T12:00:00+00:00", "run_id": run_id, "thread_id": "t-1",
            "event": event, "case_id": case_id, "customer_id": customer_id, **fields}


# ---- which entries belong to this notebook session -------------------------------------


def test_the_observer_remembers_every_run_it_observed(fresh_observer):
    first = fresh_observer.start_run("t-1", "review CUST-1001")
    second = fresh_observer.start_run("t-2", "review CUST-1042")
    assert fresh_observer.run_ids == [first, second]


def test_the_session_reads_only_its_own_runs(tmp_path, monkeypatch, fresh_observer):
    """A local trail accumulates across sessions; a Colab one does not. Be right in both."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    _write(path, [_entry("case_opened", run_id="run-old", customer_id="CUST-0001"),
                  _entry("case_opened", run_id="run-a"),
                  _entry("action_executed", run_id="run-a", action="BLOCK")])
    fresh_observer.run_ids = ["run-a"]

    entries = m.session_audit_entries()
    assert [e["event"] for e in entries] == ["case_opened", "action_executed"]
    assert all(e["run_id"] == "run-a" for e in entries)


def test_with_no_runs_recorded_the_whole_trail_is_read(tmp_path, monkeypatch,
                                                       fresh_observer):
    """Resuming a case after a kernel restart leaves no run ids — show the file instead."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    _write(path, [_entry("case_opened", run_id="run-old")])
    fresh_observer.run_ids = []
    assert len(m.session_audit_entries()) == 1


def test_a_missing_trail_reads_as_empty(tmp_path, monkeypatch, fresh_observer):
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "nothing.jsonl"))
    assert m.session_audit_entries() == []


# ---- how it reads ----------------------------------------------------------------------


def _full_case():
    return [
        _entry("case_opened", priority="P1", signal_type="FRAUD_ALERT", prior_cases=1),
        _entry("human_decision", decision="feedback", reviewed_action="MONITOR",
               feedback="Too lenient, escalate."),
        _entry("human_decision", decision="approve", reviewed_action="BLOCK"),
        _entry("action_executed", action="BLOCK", risk_score=100,
               prompt_version="v3", model="claude-haiku-4-5"),
        _entry("notification_sent", channel="push", subject="Card blocked",
               body="We paused your card.", delivery_id="ntf-1"),
    ]


def test_the_report_groups_a_case_and_names_its_outcome():
    report = m.format_audit_trail(_full_case())
    assert "CASE-1" in report and "CUST-1042" in report
    assert "FRAUD_ALERT" in report and "P1" in report
    for event in ("case_opened", "human_decision", "action_executed", "notification_sent"):
        assert event in report
    assert "BLOCK" in report


def test_the_report_shows_what_the_human_decided_and_why():
    report = m.format_audit_trail(_full_case())
    assert "feedback" in report and "approve" in report
    assert "Too lenient" in report


def test_the_report_shows_the_message_the_customer_received():
    assert "We paused your card." in m.format_audit_trail(_full_case())


def test_each_case_appears_once_with_all_its_entries():
    entries = _full_case() + [_entry("case_opened", case_id="CASE-2",
                                     customer_id="CUST-2077", priority="P3")]
    report = m.format_audit_trail(entries)
    assert report.count("CASE-1") == 1
    assert "CASE-2" in report and "CUST-2077" in report


def test_a_cancelled_case_reads_as_cancelled():
    report = m.format_audit_trail([
        _entry("case_opened", customer_id="CUST-2077"),
        _entry("human_decision", decision="reject", reviewed_action="MONITOR"),
        _entry("action_cancelled", declined_action="MONITOR")])
    assert "reject" in report and "action_cancelled" in report
    assert "action_executed" not in report


def test_an_entry_with_no_case_is_still_shown():
    """Nothing written to the trail may be silently dropped from the report."""
    report = m.format_audit_trail([_entry("notification_sent", case_id=None)])
    assert "notification_sent" in report


def test_an_empty_trail_says_so_instead_of_printing_nothing():
    report = m.format_audit_trail([])
    assert report.strip()
    assert "no" in report.lower()
