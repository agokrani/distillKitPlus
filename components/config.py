import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to custom config JSON file. If None, loads default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default_config.json"

    with open(config_path) as f:
        config = json.load(f)

    return config
