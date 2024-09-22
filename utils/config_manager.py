# utils/config_manager.py

import yaml
import os


class ConfigManager:
    _config = None

    @classmethod
    def load_config(cls, path="config/config.yaml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        with open(path, "r") as file:
            cls._config = yaml.safe_load(file)

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls.load_config()
        return cls._config
