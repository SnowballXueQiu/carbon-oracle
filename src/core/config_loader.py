import toml
import os
from typing import Any, Dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.toml")

class Config:
    _instance = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        
        with open(CONFIG_PATH, "r") as f:
            self._config_data = toml.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        # Support nested keys like "openai.model"
        keys = key.split(".")
        value = self._config_data
        try:
            for k in keys:
                if value is None:
                    return default
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global accessor
config = Config()
