import os
import yaml
import logging
from functools import wraps

def load_config(config_path="config.yaml"):
    """Load the project YAML configuration."""
    if not os.path.exists(config_path):
        # Look in parent directory if run from a subdirectory
        parent_config = os.path.join("..", config_path)
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

def setup_logging(step_name):
    """Configure basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] {step_name} - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(step_name)

def get_tcga_patient_id(barcode):
    """
    Extract the 12-character patient ID from a TCGA barcode.
    E.g., TCGA-D3-A1QA-06A-11R-A173-07 -> TCGA-D3-A1QA
    """
    if isinstance(barcode, str) and barcode.startswith("TCGA"):
        parts = barcode.split("-")
        if len(parts) >= 3:
            return "-".join(parts[0:3])
    return barcode

def is_tumor_sample(barcode):
    """
    Check if a TCGA barcode represents a tumor sample.
    01-09 are tumor, 10-19 are normal.
    """
    if isinstance(barcode, str) and barcode.startswith("TCGA"):
        parts = barcode.split("-")
        if len(parts) >= 4:
            sample_type = parts[3][:2]
            try:
                if 1 <= int(sample_type) <= 9:
                    return True
            except ValueError:
                pass
    return False

def init_paths():
    """Initialize standard project paths from config."""
    cfg = load_config()
    paths = cfg.get("paths", {})
    for k, v in paths.items():
        if k != "python":
            ensure_dir(v)
    return paths

# Quick startup action when imported
init_paths()
