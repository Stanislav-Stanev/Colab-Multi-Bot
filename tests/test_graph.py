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
    # revision guard: after MAX_REVISIONS, feedback stops looping and finalises
    assert m.route_after_review(
        {"human_decision": {"type": "feedback", "feedback": "x"},
         "revision_count": m.MAX_REVISIONS}) == "execute_action"


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
