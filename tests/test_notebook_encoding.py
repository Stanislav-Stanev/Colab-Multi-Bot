"""The generated notebook must stay UTF-8 clean.

jupytext and nbclient read and write the notebook using the process's locale encoding.
On a Windows console set to cp1251 that silently rewrites every emoji as mojibake and
then makes `jupyter execute` fail with a UnicodeDecodeError, so the corruption is worth
catching in the suite rather than in the submitted file.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "Fraud_Detection_Multi_Agent.ipynb"
SOURCE = ROOT / "fraud_multi_agent.py"

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
MOJIBAKE = re.compile(r"[рџРСÃÐÑ][-ӿ]{1,4}")


@pytest.fixture(scope="module")
def notebook_text():
    if not NOTEBOOK.exists():
        pytest.skip("notebook has not been generated yet")
    return NOTEBOOK.read_text(encoding="utf-8")


def test_source_is_utf8_clean():
    text = SOURCE.read_text(encoding="utf-8")
    assert not MOJIBAKE.search(text), "the jupytext source itself is corrupted"
    assert EMOJI.search(text)


def test_notebook_has_no_mojibake(notebook_text):
    found = MOJIBAKE.findall(notebook_text)
    assert not found, f"notebook contains mojibake, regenerate with build_notebook.py: {found[:5]}"


def test_notebook_kept_its_emoji(notebook_text):
    assert len(EMOJI.findall(notebook_text)) >= 15, "emoji were lost in generation"


@pytest.mark.parametrize("path", [SOURCE, NOTEBOOK, ROOT / "README.md"],
                         ids=["source", "notebook", "readme"])
def test_no_composite_emoji_sequences(path):
    """Only single-codepoint emoji.

    A zero-width joiner (U+200D) or variation selector (U+FE0F) renders as one glyph only
    if the reader's font ships that exact composition; otherwise they see the pieces, an
    empty box, or a stray invisible character.
    """
    if not path.exists():
        pytest.skip(f"{path.name} has not been generated yet")
    text = path.read_text(encoding="utf-8")
    assert "‍" not in text, f"{path.name} has a zero-width joiner emoji sequence"
    assert "️" not in text, f"{path.name} has a variation-selector emoji sequence"


def test_no_replacement_characters(notebook_text):
    assert "�" not in notebook_text, "notebook contains U+FFFD — characters were lost"


def test_notebook_is_valid_and_matches_the_source(notebook_text):
    nb = json.loads(notebook_text)
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for marker in ("def execute_workflow", "MemorySaver", "interrupt(", "MODEL_NAME"):
        assert marker in code, f"{marker} missing — notebook is out of date with the source"
