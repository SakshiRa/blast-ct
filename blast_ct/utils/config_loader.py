import os
from pathlib import Path

def get_config():
    """
    Returns the path to the packaged config.json.
    Assumes config is installed inside blast_ct/data/.
    """
    install_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = Path(install_dir).parent / "data" / "config.json"
    return str(config_path)
