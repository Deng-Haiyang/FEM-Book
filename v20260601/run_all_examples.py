"""Run all example JSON files and write their outputs into results/."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
RESULTS = ROOT / "results"

for model in sorted(EXAMPLES.glob("*.json")):
    output = RESULTS / f"{model.stem}_output.txt"
    print(f"Running {model.name} -> {output.relative_to(ROOT)}")
    subprocess.run(
        [sys.executable, str(SRC / "main.py"), str(model), "--output", str(output)],
        cwd=str(SRC),
        check=True,
    )
print("All examples completed.")
