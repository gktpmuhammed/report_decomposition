"""Configuration settings for the medical report decomposition system."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Input files
INPUT_FILES = {
    "train": DATA_DIR / "train_reports.csv",
}

# Output files
OUTPUT_FILES = {
    "findings": DATA_DIR / "desc_info_manual_v5.json",
    "impressions": DATA_DIR / "conc_info_manual_v5.json"
}

# Model settings
MODEL_CONFIG = {
    "name": "qwen3:8b",
    "base_url": "http://localhost:11434",
    "max_workers": 6,
    "batch_size": 250,
    "save_interval": 25,
    "chunk_size": 5000
}

# Processing settings
PROCESSING_CONFIG = {
    "cooldown": {
        "between_batches": 5,
        "between_chunks": 15
    },
    "retry": {
        "max_attempts": 2,
        "delay": 0.5
    }
}

# Logging settings
LOGGING_CONFIG = {
    "level": "DEBUG",
    "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
} 