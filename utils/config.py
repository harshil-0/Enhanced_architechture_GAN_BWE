import yaml
import os
from typing import Any, Dict

class ConfigNode:
    """Helper class to allow dot notation access to nested dictionary elements."""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            elif isinstance(value, list):
                setattr(self, key, [ConfigNode(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert namespace back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in value]
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return str(self.to_dict())


def load_config(config_path: str) -> ConfigNode:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the yaml config file.
        
    Returns:
        ConfigNode: Dot-accessible configuration tree.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        
    return ConfigNode(data)
