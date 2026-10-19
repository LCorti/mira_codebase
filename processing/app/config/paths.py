import os
from pathlib import Path

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR.joinpath("config")
LOGS_DIR = ROOT_DIR.joinpath("logs")
