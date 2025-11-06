import yaml
import os

def load_config(config_path: str = "config/storage_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
