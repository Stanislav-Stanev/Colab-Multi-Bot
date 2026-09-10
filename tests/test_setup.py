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
