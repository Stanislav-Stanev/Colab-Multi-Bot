"""Guards on the request we send to Anthropic.

These lock in the fixes for two failure modes that only show up on a live call:
sampling parameters are rejected with a 400 on the Claude 5 models, and LangChain's
default structured-output method uses forced tool calling.
"""
import pathlib

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


def test_model_and_max_tokens(llm):
    payload = llm._get_request_payload(MESSAGES)
    assert payload["model"] == m.MODEL_NAME
    assert payload["max_tokens"] == 16000


def test_effort_is_only_sent_when_the_model_supports_it(llm):
    """Haiku 4.5 answers a request carrying `effort` with a 400, so it must be omitted."""
    payload = llm._get_request_payload(MESSAGES)
    if m.EFFORT is None:
        assert "effort" not in payload.get("output_config", {})
    else:
        assert payload["output_config"]["effort"] == m.EFFORT


def test_every_model_in_pricing_is_priced():
    assert m.MODEL_NAME in m.PRICING, "unknown model would report a $0.00 cost"
    for name, (price_in, price_out) in m.PRICING.items():
        assert price_in > 0 and price_out > price_in, name


def test_the_observer_is_registered_as_a_callback(llm):
    """Token usage is captured without any call site reporting it."""
    assert m.OBS in (llm.callbacks or []), "the LLM client must report usage to the observer"


def test_observer_survives_a_response_without_usage():
    observer = m.WorkflowObserver()
    observer.on_llm_end(type("R", (), {"generations": [[object()]]})())
    assert observer.calls == 0


def test_structured_output_uses_native_json_schema_not_forced_tools(llm):
    bound = m.structured(m.AnalystNarrative)
    inner = getattr(bound, "first", bound)
    payload = llm._get_request_payload(MESSAGES, **getattr(inner, "kwargs", {}))

    output_config = payload["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    if m.EFFORT is not None:
        assert output_config["effort"] == m.EFFORT      # effort survives the merge
    assert payload.get("tool_choice") is None           # never forced tool calling
    assert "temperature" not in payload


def test_tool_binding_sends_the_analysts_kit_unforced(llm):
    """Only what the analyst is trusted to call reaches the API (review F10)."""
    payload = llm._get_request_payload(MESSAGES, **llm.bind_tools(m.TOOLS).kwargs)
    assert {t["name"] for t in payload["tools"]} == {
        "fetch_customer_transactions", "calculate_risk_score"}
    assert payload.get("tool_choice") is None
    assert "temperature" not in payload


def test_llm_client_is_cached(llm):
    assert m.get_llm() is llm


# ---- prompt caching (review F2) --------------------------------------------------------


def test_a_system_prompt_is_sent_as_a_cacheable_block():
    """The agent prompts are frozen for the life of a run — exactly the cacheable shape.

    A cached prefix is read at a fraction of the input price, and the saving grows with
    the model: what is cents on Haiku is not on Opus.
    """
    block = m.cacheable_system(m.TRIAGE_PROMPT)
    assert isinstance(block.content, list)
    assert block.content[0]["type"] == "text"
    assert block.content[0]["text"] == m.TRIAGE_PROMPT
    assert block.content[0]["cache_control"] == {"type": "ephemeral"}


def test_the_cached_prefix_carries_no_per_case_content():
    """Anything that varies per case must sit after the breakpoint, or nothing ever hits.

    A customer id or a timestamp inside the cached block changes the prefix on every
    call, and a cache that never hits costs more than no cache at all.
    """
    for prompt in (m.TRIAGE_PROMPT, m.FRAUD_ANALYST_PROMPT, m.COMPLIANCE_OFFICER_PROMPT,
                   m.CUSTOMER_COMMS_PROMPT):
        assert "CUST-" not in prompt
        assert "{" not in prompt, "an f-string style placeholder would vary the prefix"


def test_every_agent_sends_its_prompt_through_the_cacheable_helper():
    """A prompt built inline would silently opt that agent out of caching."""
    source = pathlib.Path(m.__file__).read_text(encoding="utf-8")
    agent_prompts = ("TRIAGE_PROMPT", "FRAUD_ANALYST_PROMPT", "COMPLIANCE_OFFICER_PROMPT",
                     "CUSTOMER_COMMS_PROMPT")
    for name in agent_prompts:
        assert f"SystemMessage(content={name})" not in source, \
            f"{name} is sent uncached — wrap it in cacheable_system()"


def test_usage_reporting_counts_cache_reads(fresh_observer):
    """A cache that is not measured is a cache nobody knows is broken."""
    class _Resp:
        def __init__(self):
            message = type("M", (), {"usage_metadata": {
                "input_tokens": 120, "output_tokens": 8,
                "input_token_details": {"cache_read": 900, "cache_creation": 0}}})()
            self.generations = [[type("G", (), {"message": message})()]]

    fresh_observer.current_node = "triage"
    fresh_observer.on_llm_end(_Resp())

    assert fresh_observer.nodes["triage"].cached_tokens == 900
    assert "cache" in fresh_observer.report().lower()
