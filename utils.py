from easydict import EasyDict as edict
import yaml
from datetime import datetime
import os


def load_config(config_path: str) -> edict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return edict(config)


def dump_config(config: edict, config_path: str) -> edict:
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def run_name_from_config(config: edict) -> str:
    formatted_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return "-".join(
        [
            f"{config.io.run_tag}",
            f"{config.model.adapter}",
            f"dataset-{config.io.dataset_type}",
            f"batch-size-{config.training.batch_size}",
            f"epochs-{config.training.num_train_epochs}",
            f"{formatted_datetime}",
        ]
    )


def dataset_name_from_dataset_type(dataset_type: str) -> str:
    if dataset_type == "science":
        return "vic0428/imdb-card-pred-science"
    elif dataset_type == "binary":
        return "vic0428/imdb-card-pred-binary"
    elif dataset_type == "decimal":
        return "vic0428/imdb-card-pred-decimal"
    raise ValueError(f"Unknown dataset type: {dataset_type}")
