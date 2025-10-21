import os
import json
import logging
from datetime import datetime
from pathlib import Path

# --- Logging Setup ---


def get_logger(name: str = "retail_churn") -> logging.Logger:
    """
    Creates and returns a logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# --- File Handling ---


def ensure_dir(path: str):
    """
    Ensures that a directory exists.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filepath: str):
    """
    Saves a dictionary to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """
    Loads a JSON file into a dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

# --- Timestamping ---


def get_timestamp() -> str:
    """
    Returns a timestamp string for logging or filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Model File Path Helper ---


def get_model_path(model_name: str = "churn_model.pkl") -> Path:
    """
    Returns the absolute path to a saved model file.
    """
    return Path(__file__).resolve().parent.parent / "model" / model_name


# Example usage
if __name__ == "__main__":
    logger = get_logger()
    logger.info("Utility functions working fine.")
    ensure_dir("logs")
    save_json({"status": "ok"}, "logs/test.json")
    print(load_json("logs/test.json"))
