import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("TRIAGENT_DB", str(tmp_path / "test.sqlite"))
    from triagent import config, service

    config.get_settings.cache_clear()
    service.get_service.cache_clear()
    if not config.get_settings().model_path.exists():
        pytest.skip("train the model first: python scripts/train.py")
    from triagent.api import app

    yield TestClient(app)
    service.get_service.cache_clear()
    config.get_settings.cache_clear()
    os.environ.pop("TRIAGENT_DB", None)


def test_ticket_lifecycle(client):
    r = client.post("/tickets", json={"text": "Kartım çalındı, TC 10000000146, kapatın"})
    assert r.status_code == 200
    t = r.json()
    assert "10000000146" not in t["masked_text"] and "[TCKN]" in t["masked_text"]
    assert t["priority"] == "P1" and t["status"] == "pending_review"

    r = client.post(f"/tickets/{t['ticket_id']}/review", json={"action": "approve"})
    assert r.json()["status"] == "approved"

    r = client.post(f"/tickets/{t['ticket_id']}/review", json={"action": "approve"})
    assert r.status_code == 409  # cannot review twice