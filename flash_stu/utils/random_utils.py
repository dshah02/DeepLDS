import logging
import yaml
import os

def get_logger(name=__name__):
    """
    Returns a logger with the specified name.
    Configures the logger with a standard format and level.
    
    Args:
        name (str): The name of the logger, typically __name__ for the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers in Jupyter environments or repeated imports
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Set the default log level here
    return logger


def save_yaml_config(config_dict, output_dir, yaml_filename):
    with open(os.path.join(output_dir, yaml_filename), "w") as file:
        yaml.safe_dump(config_dict, file, default_flow_style=False)
