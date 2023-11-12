from easydict import EasyDict as edict
from yaml import safe_load

def load_config(config_path: str) -> edict:
    with open(config_path, "r") as f:
        config = safe_load(f)
    return edict(config)