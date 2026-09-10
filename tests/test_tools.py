import json

import fraud_multi_agent as m


def test_clean_customer_low_score():
    txs = m._fetch("CUST-1001")["transactions"]
    result = m._score(txs)
    assert result["risk_score"] < 40 and result["triggered_rules"] == []


def test_card_testing_customer_high_score():
    txs = m._fetch("CUST-1042")["transactions"]
    result = m._score(txs)
    assert result["risk_score"] >= 70
    assert "card_testing" in result["triggered_rules"]


def test_geo_anomaly_detected():
    txs = m._fetch("CUST-2077")["transactions"]
    assert "geo_anomaly" in m._score(txs)["triggered_rules"]


def test_velocity_detected():
    txs = m._fetch("CUST-1337")["transactions"]
    assert "velocity" in m._score(txs)["triggered_rules"]


def test_amount_outlier_detected():
    txs = m._fetch("CUST-3050")["transactions"]
    assert "amount_outlier" in m._score(txs)["triggered_rules"]


def test_unknown_customer_error():
    out = m._fetch("CUST-9999")
    assert out.get("error")
    assert "CUST-1001" in out["known_ids"]


def test_customer_id_is_case_insensitive():
    assert m._fetch("cust-1001")["customer"]["customer_id"] == "CUST-1001"


def test_empty_transactions_score_zero():
    assert m._score([])["risk_score"] == 0


def test_tools_are_langchain_tools():
    names = {t.name for t in m.TOOLS}
    assert names == {"fetch_customer_transactions", "calculate_risk_score", "check_sanctions_list"}
    payload = json.loads(m.fetch_customer_transactions.invoke({"customer_id": "CUST-1001"}))
    assert payload["customer"]["customer_id"] == "CUST-1001"


def test_calculate_risk_score_tool_accepts_wrapper_and_array():
    fetched = m.fetch_customer_transactions.invoke({"customer_id": "CUST-1042"})
    from_wrapper = json.loads(m.calculate_risk_score.invoke({"transactions_json": fetched}))
    txs_only = json.dumps(json.loads(fetched)["transactions"])
    from_array = json.loads(m.calculate_risk_score.invoke({"transactions_json": txs_only}))
    assert from_wrapper == from_array
    assert from_wrapper["risk_score"] >= 70


def test_calculate_risk_score_tool_reports_bad_input():
    out = json.loads(m.calculate_risk_score.invoke({"transactions_json": "not json"}))
    assert "error" in out


def test_sanctions_hit_and_miss():
    assert json.loads(
        m.check_sanctions_list.invoke({"customer_name": "Viktor Baranov"}))["match"] is True
    assert json.loads(
        m.check_sanctions_list.invoke({"customer_name": "Maria Petrova"}))["match"] is False
