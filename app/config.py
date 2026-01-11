"""Configuration management for the RFSoC AWG application."""

import json
import os
from typing import Any


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


class Config:
    """Configuration class that loads and manages app settings."""
    
    def __init__(self):
        self._data: dict = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from config.json."""
        with open(CONFIG_FILE, "r") as f:
            self._data = json.load(f)
    
    def save(self) -> None:
        """Save configuration to config.json."""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._data, f, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save."""
        self._data[key] = value
        self.save()


# Global config instance - loaded on import
config = Config()
