import fraud_multi_agent as m


def test_module_imports_without_llm_calls():
    assert m.SKIP_DEMOS is True
    assert m.MODEL_NAME.startswith("claude")


def test_get_secret_reads_env(monkeypatch):
    monkeypatch.setenv("MY_TEST_SECRET", "xyz")
    assert m.get_secret("MY_TEST_SECRET") == "xyz"


def test_get_secret_missing_raises_actionable():
    try:
        m.get_secret("DOES_NOT_EXIST_123")
        assert False, "should raise"
    except RuntimeError as e:
        assert "Colab" in str(e) and ".env" in str(e)


def test_agents_carry_the_revolutbank_persona():
    """Every agent speaks as RevolutBank Fraud Operations (rebrand, plan §2)."""
    for prompt in (m.FRAUD_ANALYST_PROMPT, m.COMPLIANCE_OFFICER_PROMPT):
        assert "RevolutBank" in prompt
        assert "Fraud Operations" in prompt


def test_tools_read_as_revolutbank_integrations():
    """Tool descriptions name the RevolutBank systems they mock (plan §2)."""
    assert "RevolutBank Core Banking" in m.fetch_customer_transactions.description
    assert "RevolutBank Screening" in m.check_sanctions_list.description
