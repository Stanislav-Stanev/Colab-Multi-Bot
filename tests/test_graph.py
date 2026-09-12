import fraud_multi_agent as m


def test_extract_customer_id():
    assert m._extract_customer_id("please check cust-1042 for fraud") == "CUST-1042"
    assert m._extract_customer_id("review CUST 2077 today") == "CUST-2077"
    assert m._extract_customer_id("what was the score again?") is None


def test_no_customer_node_lists_the_known_ids_without_calling_an_llm():
    out = m.no_customer({"user_request": "is anything suspicious?"})
    assert "CUST-1042" in out["final_output"]
    assert "customer id" in out["final_output"].lower()


def test_route_after_review():
    assert m.route_after_review(
        {"human_decision": {"type": "approve"}, "revision_count": 0}) == "execute_action"
    assert m.route_after_review(
        {"human_decision": {"type": "feedback", "feedback": "x"},
         "revision_count": 1}) == "compliance_officer"
    assert m.route_after_review(
        {"human_decision": {"type": "reject"}, "revision_count": 0}) == "cancel_action"
    # revision guard: after MAX_REVISIONS the loop stops, but it must stop *closed* —
    # the reviewer's last word was an objection, so the contested action is never executed.
    assert m.route_after_review(
        {"human_decision": {"type": "feedback", "feedback": "x"},
         "revision_count": m.MAX_REVISIONS}) == "escalate"


def test_exhausting_the_revision_budget_never_executes_the_contested_action():
    """Fail closed, not open (review F1).

    A loop breaker that finalises the action the human was still arguing with performs a
    card block nobody approved. Exhaustion parks the case for a senior reviewer instead.
    """
    for count in (m.MAX_REVISIONS, m.MAX_REVISIONS + 5):
        assert m.route_after_review(
            {"human_decision": {"type": "feedback", "feedback": "still wrong"},
             "revision_count": count}) == "escalate"


def test_escalate_parks_the_case_without_acting(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    out = m.escalate({"customer_id": "CUST-1337", "case_id": "CASE-ESC",
                      "recommended_action": "BLOCK", "report": "# Report",
                      "revision_count": m.MAX_REVISIONS})
    assert "NOT EXECUTED" in out["final_output"]
    assert "senior" in out["final_output"].lower()
    # the report the reviewer kept objecting to is preserved for whoever picks the case up
    assert "# Report" in out["final_output"]

    import json
    entries = [json.loads(line) for line
               in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    parked = [e for e in entries if e["event"] == "revision_budget_exhausted"]
    assert len(parked) == 1
    assert parked[0]["declined_action"] == "BLOCK"
    assert parked[0]["revisions"] == m.MAX_REVISIONS


def test_execute_action_renders_block_effect():
    out = m.execute_action({"recommended_action": "BLOCK", "customer_id": "CUST-1042",
                            "report": "# Report"})
    assert "BLOCKED" in out["final_output"] and "# Report" in out["final_output"]


def test_cancel_action_reports_no_change():
    out = m.cancel_action({"customer_id": "CUST-2077"})
    assert "CANCELLED" in out["final_output"] and "CUST-2077" in out["final_output"]


def test_graph_compiles_with_memory():
    assert m.graph is not None
    assert m.graph.checkpointer is m.checkpointer
    nodes = set(m.graph.get_graph().nodes)
    for n in ["triage", "fraud_analyst", "compliance_officer", "comms_draft",
              "human_review", "execute_action", "cancel_action", "comms_deliver",
              "followup_qa", "no_customer", "customer_not_found"]:
        assert n in nodes
