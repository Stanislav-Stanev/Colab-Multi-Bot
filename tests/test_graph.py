import fraud_multi_agent as m


def test_extract_customer_id():
    assert m._extract_customer_id("please check cust-1042 for fraud") == "CUST-1042"
    assert m._extract_customer_id("what was the score again?") is None


def test_route_after_intake():
    assert m.route_after_intake({"customer_id": "CUST-1001", "messages": []}) == "fraud_analyst"
    assert m.route_after_intake({"customer_id": None, "messages": ["prior"]}) == "followup_qa"
    assert m.route_after_intake({"customer_id": None, "messages": []}) == "fraud_analyst"


def test_route_after_review():
    assert m.route_after_review(
        {"human_decision": {"type": "approve"}, "revision_count": 0}) == "execute_action"
    assert m.route_after_review(
        {"human_decision": {"type": "feedback", "feedback": "x"},
         "revision_count": 1}) == "compliance_officer"
    assert m.route_after_review(
        {"human_decision": {"type": "reject"}, "revision_count": 0}) == "cancel_action"
    # revision guard: after MAX_REVISIONS, feedback stops looping and finalises
    assert m.route_after_review(
        {"human_decision": {"type": "feedback", "feedback": "x"},
         "revision_count": m.MAX_REVISIONS}) == "execute_action"


def test_intake_populates_state():
    out = m.intake({"user_request": "review CUST-2077 please"})
    assert out["customer_id"] == "CUST-2077"
    assert out["revision_count"] == 0
    assert out["messages"][0].content == "review CUST-2077 please"


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
    for n in ["intake", "fraud_analyst", "compliance_officer", "human_review",
              "execute_action", "cancel_action", "followup_qa"]:
        assert n in nodes
