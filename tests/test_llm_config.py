"""Guards on the request we send to Anthropic.

These lock in the fixes for two failure modes that only show up on a live call:
sampling parameters are rejected with a 400 on the Claude 5 models, and LangChain's
default structured-output method uses forced tool calling.
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import fraud_multi_agent as m


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-no-request-is-made")
    m._llm_cache.clear()
    client = m.get_llm()
    yield client
    m._llm_cache.clear()


MESSAGES = [SystemMessage(content="sys"), HumanMessage(content="hi")]


def test_no_sampling_parameters_are_sent(llm):
    payload = llm._get_request_payload(MESSAGES)
    for removed in ("temperature", "top_p", "top_k"):
        assert removed not in payload, f"{removed} returns a 400 on the Claude 5 models"


def test_model_and_effort(llm):
    payload = llm._get_request_payload(MESSAGES)
    assert payload["model"] == m.MODEL_NAME
    assert payload["output_config"]["effort"] == m.EFFORT
    assert payload["max_tokens"] == 16000


def test_structured_output_uses_native_json_schema_not_forced_tools(llm):
    bound = m.structured(m.RiskAssessment)
    inner = getattr(bound, "first", bound)
    payload = llm._get_request_payload(MESSAGES, **getattr(inner, "kwargs", {}))

    output_config = payload["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["effort"] == m.EFFORT          # effort survives the merge
    assert payload.get("tool_choice") is None           # never forced tool calling
    assert "temperature" not in payload


def test_tool_binding_sends_all_three_tools_unforced(llm):
    payload = llm._get_request_payload(MESSAGES, **llm.bind_tools(m.TOOLS).kwargs)
    assert {t["name"] for t in payload["tools"]} == {
        "fetch_customer_transactions", "calculate_risk_score", "check_sanctions_list"}
    assert payload.get("tool_choice") is None
    assert "temperature" not in payload


def test_llm_client_is_cached(llm):
    assert m.get_llm() is llm
