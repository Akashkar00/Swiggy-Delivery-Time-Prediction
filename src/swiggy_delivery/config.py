from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "swiggy.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "swiggy_cleaned.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
