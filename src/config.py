"""
Configuration loading and validation utilities.
"""

import os
import yaml


def load_configuration(config_path):
  """
    Load and validate the configuration file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
  try:
    with open(config_path, 'r') as f:
      config = yaml.safe_load(f)
    print(f"Configuration loaded from: {config_path}")
    return config
  except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
  except yaml.YAMLError as e:
    raise yaml.YAMLError(f"Error parsing configuration file: {e}")


def validate_config(config):
  """
    Validate the loaded configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
  required_keys = ['agent', 'env', 'device']
  for key in required_keys:
    if key not in config:
      raise ValueError(f"Missing required configuration key: {key}")

  if 'name' not in config['agent']:
    raise ValueError("Missing required 'name' in agent configuration")
