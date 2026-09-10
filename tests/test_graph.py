import fraud_multi_agent as m


def test_extract_customer_id():
    assert m._extract_customer_id("please check cust-1042 for fraud") == "CUST-1042"
    assert m._extract_customer_id("review CUST 2077 today") == "CUST-2077"
    assert m._extract_customer_id("what was the score again?") is None


def test_route_after_intake_starts_an_investigation_when_a_customer_is_named():
    assert m.route_after_intake(
        {"customer_id": "CUST-1001", "messages": ["the request"]}) == "fraud_analyst"


def test_route_after_intake_answers_from_memory_only_with_prior_history():
    # intake has already appended the current request, so a fresh thread holds exactly one
    # message; anything more means the thread carries a previous investigation.
    assert m.route_after_intake(
        {"customer_id": None, "messages": ["earlier turn", "the request"]}) == "followup_qa"


def test_route_after_intake_asks_for_a_customer_id_on_a_fresh_thread():
    assert m.route_after_intake(
        {"customer_id": None, "messages": ["the request"]}) == "no_customer"
    assert m.route_after_intake({"customer_id": None, "messages": []}) == "no_customer"


def test_no_customer_node_lists_the_known_ids_without_calling_an_llm():
    out = m.no_customer({"user_request": "is anything suspicious?"})
    assert "CUST-1042" in out["final_output"]
    assert "customer id" in out["final_output"].lower()


def test_intake_resets_a_previous_case_on_the_same_thread():
    stale = {"user_request": "now review CUST-2077", "revision_count": 3,
             "report": "old report", "risk_assessment": {"risk_score": 95},
             "recommended_action": "BLOCK", "final_output": "old output",
             "human_decision": {"type": "feedback", "feedback": "old"}}
    out = m.intake(stale)
    assert out["customer_id"] == "CUST-2077"
    assert out["revision_count"] == 0
    for cleared in ("report", "risk_assessment", "recommended_action",
                    "final_output", "human_decision"):
        assert out[cleared] is None, f"{cleared} leaked from the previous investigation"


def test_intake_keeps_history_for_a_follow_up_question():
    out = m.intake({"user_request": "what was the score again?"})
    assert out["customer_id"] is None
    assert "revision_count" not in out       # a follow-up must not reset the case


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
