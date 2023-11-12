from easydict import EasyDict as edict
from yaml import safe_load
from datetime import datetime


def load_config(config_path: str) -> edict:
    with open(config_path, "r") as f:
        config = safe_load(f)
    return edict(config)


def run_name_from_config(config: edict) -> str:
    formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{config.io.run_tag}-{config.adaptor}-dataset-{config.training.dataset_type}-batch-size-{config.training_batch_size}-epochs-{config.training.num_train_epochs}-{formatted_datetime}"


def dataset_name_from_dataset_type(dataset_type: str) -> str:
    if dataset_type == "science":
        return "vic0428/imdb-card-pred-science"
    elif dataset_type == "binary":
        return "vic0428/imdb-card-pred-binary"
    elif dataset_type == "decimal":
        return "vic0428/imdb-card-pred-decimal"
    raise ValueError(f"Unknown dataset type: {dataset_type}")
