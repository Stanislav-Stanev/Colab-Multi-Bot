"""Regenerate (and optionally run) the submission notebook from fraud_multi_agent.py.

Use this instead of calling jupytext/jupyter directly. On a Windows console whose locale
is not UTF-8 (cp1251, cp1252, ...), those tools read and write the notebook in the legacy
code page, which turns every emoji into mojibake (🔎 becomes "рџ”Ћ") and later makes
`jupyter execute` fail outright. This script forces UTF-8 mode for the child processes
and refuses to leave a corrupted notebook behind.

    python build_notebook.py            # regenerate only
    python build_notebook.py --run      # regenerate, then execute with a live API key
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "fraud_multi_agent.py"
NOTEBOOK = ROOT / "Fraud_Detection_Multi_Agent.ipynb"

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
# UTF-8 bytes decoded as a Cyrillic/Latin-1 code page leave these tell-tale clusters.
MOJIBAKE = re.compile(r"[рџРСÃÐÑ][-ӿ]{1,4}")


def utf8_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"          # make open() default to UTF-8 in the child process
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(*args: str) -> None:
    result = subprocess.run([sys.executable, "-m", *args], env=utf8_env(), cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"FAILED: {' '.join(args)}")


def check_encoding() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    emoji, mojibake = len(EMOJI.findall(text)), len(MOJIBAKE.findall(text))
    if mojibake or not emoji:
        sys.exit(f"Notebook is mojibake-corrupted ({emoji} emoji, {mojibake} bad sequences). "
                 f"Regenerate it with this script rather than jupytext directly.")
    print(f"OK: notebook encoding is clean ({emoji} emoji, no mojibake)")


def main() -> None:
    run("jupytext", "--to", "ipynb", "--set-kernel", "python3",
        str(SOURCE), "-o", str(NOTEBOOK))
    check_encoding()

    if "--run" in sys.argv:
        print("Executing the notebook (needs ANTHROPIC_API_KEY in .env)...")
        run("jupyter", "execute", "--inplace", "--allow-errors",
            "--timeout=900", str(NOTEBOOK))
        check_encoding()
        print("Done. Outputs are saved in the notebook.")


if __name__ == "__main__":
    main()
