# config.py
from pathlib import Path
import platform

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# MLmodel directory
MODEL_DIR = BASE_DIR / "MLmodel"

# Model paths
CROP_MODEL_PATH = MODEL_DIR / "Cropbest_v1.0.pt"
FACE_MODEL_PATH = MODEL_DIR / "Facebest_v1.0.pt"
PAN_MODEL_PATH = MODEL_DIR / "Pancardbest_v1.1.pt"
AADHAAR_MASK_MODEL_PATH = MODEL_DIR / "AadhaarMaskbest_v1.2.pt"

# Tesseract path
TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if platform.system() == "Windows"
    else "tesseract"
)
