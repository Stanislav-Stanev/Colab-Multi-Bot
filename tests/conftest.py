import os
import sys
import tempfile

os.environ["FRAUD_SKIP_DEMOS"] = "1"  # must be set BEFORE fraud_multi_agent is imported
# Keep run artefacts (audit trail, checkpoint database) out of the repo during tests.
_ARTEFACTS = tempfile.mkdtemp()
os.environ["FRAUD_AUDIT_LOG"] = os.path.join(_ARTEFACTS, "audit_log.jsonl")
os.environ["FRAUD_CHECKPOINT_DB"] = os.path.join(_ARTEFACTS, "cases.sqlite")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest  # noqa: E402 - must follow the sys.path/env setup above

import fraud_multi_agent as m  # noqa: E402


@pytest.fixture
def fresh_observer(monkeypatch):
    """A test-local observer, so token counts and events never leak between tests.

    Which node is being attributed lives in a ContextVar rather than on the observer (so
    branches running on worker threads keep their own), and a ContextVar outlives the
    object — so it is reset here too, or one test's node would label the next one's events.
    """
    observer = m.WorkflowObserver()
    monkeypatch.setattr(m, "OBS", observer)
    token = m._CURRENT_NODE.set(None)
    yield observer
    m._CURRENT_NODE.reset(token)
