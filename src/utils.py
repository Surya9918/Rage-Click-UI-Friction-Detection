import logging
import yaml
from pathlib import Path

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
