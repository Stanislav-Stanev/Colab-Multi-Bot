"""Guards on the acceptance test of the whole project (plan §0).

> Upload the notebook to Colab, add ANTHROPIC_API_KEY, press Runtime → Run all, and it
> runs top to bottom unattended.

These tests fail if a later change quietly breaks that: a cell that waits for input, a
Drive mount, a second install cell, or a dependency with no fallback.
"""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "fraud_multi_agent.py"
NOTEBOOK = ROOT / "Fraud_Detection_Multi_Agent.ipynb"


@pytest.fixture(scope="module")
def code_cells() -> list:
    if not NOTEBOOK.exists():
        pytest.skip("notebook has not been generated yet")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]
            if cell["cell_type"] == "code"]


def test_exactly_one_install_cell(code_cells):
    """One pip cell at the top is the only setup a reviewer should have to trust."""
    installs = [c for c in code_cells if "pip install" in c]
    assert len(installs) == 1, f"expected one install cell, found {len(installs)}"
    assert code_cells.index(installs[0]) == 0, "the install cell must come first"


def _executable(cell: str) -> str:
    """The cell without its comment lines — a comment cannot install anything."""
    return "\n".join(line for line in cell.splitlines()
                     if not line.lstrip().startswith("#") or line.lstrip().startswith("#%")
                     or line.lstrip().startswith("# %pip"))


def test_the_install_cell_needs_nothing_but_pip(code_cells):
    install = _executable(next(c for c in code_cells if "pip install" in c))
    for forbidden in ("apt-get", "apt install", "!wget", "!curl", "conda ", "docker",
                      "systemctl"):
        assert forbidden not in install, f"{forbidden!r} is not available in Colab"


def test_no_cell_blocks_on_interactive_input(code_cells):
    """`Run all` must never stop waiting for a human to type something.

    `ask_human` may *define* an `input()` path for a person driving the workflow by hand;
    what must not happen is a demo cell calling it.
    """
    for index, cell in enumerate(code_cells):
        if "def ask_human" in cell:
            continue                      # the definition, not a call
        assert not re.search(r"(?<![\w.])input\s*\(", cell), \
            f"code cell {index} calls input() and would stall Run all"
        assert "getpass" not in cell, f"code cell {index} prompts for a secret"


def test_nothing_mounts_google_drive_during_a_run(code_cells):
    """Mounting opens an OAuth pop-up, which stalls Run all (plan §0)."""
    for index, cell in enumerate(code_cells):
        assert "drive.mount" not in cell, f"code cell {index} mounts Drive"


def test_drive_is_opt_in_and_off():
    source = SOURCE.read_text(encoding="utf-8")
    assert "USE_DRIVE = False" in source


def test_every_optional_import_has_a_fallback():
    """An import that can fail must not be able to fail the run."""
    source = SOURCE.read_text(encoding="utf-8")
    for optional in ("from langgraph.checkpoint.sqlite import SqliteSaver",
                     "from google.colab import userdata",
                     "from IPython.display import Image, display",
                     "from dotenv import load_dotenv"):
        assert optional in source, f"{optional} moved — check its fallback"
        before = source.split(optional)[0]
        # the nearest enclosing block above the import must be a try
        assert before.rstrip().endswith("try:") or "try:" in before.rsplit("\n\n", 1)[-1], \
            f"{optional} is not guarded by try/except"


def test_only_the_anthropic_key_is_required():
    """Any other secret must be optional, or Run all needs setup the plan forbids."""
    source = SOURCE.read_text(encoding="utf-8")
    required = set(re.findall(r"get_secret\(\s*[\"']([A-Z_]+)[\"']\s*\)", source))
    assert required == {"ANTHROPIC_API_KEY"}, f"extra required secrets: {required}"
    optional = set(re.findall(r"get_secret_or_none\(\s*[\"']([A-Z_]+)[\"']\s*\)", source))
    assert "LANGSMITH_API_KEY" in optional


def test_the_demo_cells_script_every_human_decision(code_cells):
    """Each run_scenario that reaches the gate carries its decisions, or Run all hangs."""
    scenario_cells = [c for c in code_cells if "run_scenario(" in c]
    assert len(scenario_cells) >= 5, "the assignment needs at least five scenarios"
    for cell in scenario_cells:
        if "CUST-9999" in cell:
            continue                      # stops before the gate: no decision needed
        assert "decisions=" in cell, f"scenario without scripted decisions:\n{cell[:120]}"


def test_no_external_network_calls_beyond_the_model_api():
    source = SOURCE.read_text(encoding="utf-8")
    for forbidden in ("requests.get(", "requests.post(", "urlopen(", "httpx.get("):
        assert forbidden not in source, f"{forbidden} reaches a host Colab may not allow"


def test_artefacts_are_written_to_relative_paths():
    """Absolute paths would break on a runtime whose filesystem differs."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'os.getenv("FRAUD_AUDIT_LOG", "audit_log.jsonl")' in source
    assert 'os.getenv("FRAUD_CHECKPOINT_DB", "revolutbank_cases.sqlite")' in source


def test_the_notebook_ends_by_printing_the_run_s_audit_trail(code_cells):
    """The last thing a reviewer scrolls to is the record of what happened."""
    last = code_cells[-1]
    assert "format_audit_trail" in last and "session_audit_entries" in last
    assert "verify_audit_trail" in last, "the trail is printed without being vouched for"
