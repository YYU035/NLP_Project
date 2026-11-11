"""Repository resources and writable state, independent of the caller's cwd.

Source checkout installation is editable. RESEARCH_VERIFIER_ROOT can select a
different resource checkout; RESEARCH_VERIFIER_DATA_DIR can select model storage.
Neither environment variable is accepted from uploaded data.
"""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("RESEARCH_VERIFIER_ROOT", PACKAGE_ROOT.parents[1])).resolve()
DATA_DIR = Path(os.environ.get("RESEARCH_VERIFIER_DATA_DIR", PROJECT_ROOT / "data")).resolve()
MODELS_DIR = DATA_DIR / "models"
RETRIEVAL_MODEL_DIR = MODELS_DIR / "retrieval_encoder"
EVIDENCE_MODEL_DIR = MODELS_DIR / "evidence_encoder"
VERIFIER_MODEL_DIR = MODELS_DIR / "claim_verifier"
LANGUAGE_MODEL_DIR = MODELS_DIR / "local_language_model"
CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
CONFIG_DIR = PROJECT_ROOT / "config"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
APP_PATH = PACKAGE_ROOT / "ui" / "app.py"
