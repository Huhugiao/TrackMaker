"""Project path helpers for models and generated outputs."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = MODELS_DIR
# Evaluation artifacts now write directly under outputs/ instead of outputs/eval/.
EVAL_OUTPUT_DIR = OUTPUTS_DIR


def ensure_output_dirs() -> None:
    for path in (MODELS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
