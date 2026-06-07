import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load models.yaml configuration.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Dictionary with 'models' key containing list of model configs
    """

    config_path = Path(config_dir) / "models.yaml"
    return load_yaml_config(str(config_path))


def get_model_by_name(model_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """Get specific model configuration by name.
    
    Args:
        model_name: Name of the model to retrieve
        config_dir: Directory containing config files
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model not found in configuration
    """

    config = load_model_config(config_dir)
    models = config.get("models", [])
    
    for model_config in models:
        if model_config.get("name") == model_name:
            return model_config
    
    raise ValueError(f"Model '{model_name}' not found in configuration")
