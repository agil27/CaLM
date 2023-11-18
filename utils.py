from easydict import EasyDict as edict
import yaml
from datetime import datetime
import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import numpy as np
from transformers import TrainingArguments


def create_if_not_exists(filename: str):
    dirname = os.path.dirname(filename)
    if dirname == "":
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_config(config_path: str) -> edict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return edict(config)


def dump_config(config: edict, config_path: str) -> edict:
    create_if_not_exists(config_path)
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


def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cuda:0") -> tuple:
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir, device_map=device, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


def populate_default_arguments_for_config(config: edict) -> edict:
    # Default arguments
    if "max_seq_length" not in config.io:
        config.io.max_seq_length = 512
    if "num_train_epochs" not in config.training:
        config.training.num_train_epochs = 3
    if "max_steps" not in config.training:
        config.training.max_steps = -1

    # Run related arguments
    config.io.run_name = run_name_from_config(config)
    config.io.run_output_dir = os.path.join(config.io.output_dir, config.io.run_name)
    if not os.path.exists(config.io.run_output_dir):
        os.makedirs(config.io.run_output_dir)

    return config


def load_training_args(config: edict) -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=config.io.run_output_dir,
        do_train=True,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=config.training.optimizer,
        logging_steps=10,
        learning_rate=config.training.learning_rate,
        bf16=config.model.use_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_total_limit=5,
    )

    training_args = training_args.set_dataloader(
        train_batch_size=config.training.batch_size
    )

    return training_args


class MetricLogger:
    """
    A logger that logs metrics and losses, as well as the ground truths and predictions in a CSV format.
    It can print out the metrics and running stats.
    """

    def __init__(self, log_file: str, metric_name: str = "metric"):
        self.log_file = log_file
        create_if_not_exists(self.log_file)

        with open(self.log_file, "a+") as f:
            f.write("truth,pred,%s\n" % (metric_name,))

        self.metric_name = metric_name
        self.metrics = []
        self.losses = []

    def log_and_return_metric(self, truth, pred, metric):
        self.metrics.append(metric)

        log = "%d,%d,%.5f\n" % (truth, pred, metric)

        with open(self.log_file, "a+") as f:
            f.write(log)

        return log

    def print_running_stats(self, data, data_name):
        print("### Running stats for %s ###" % data_name)
        print("mean:", np.mean(data))
        print("std:", np.std(data))
        print("median:", np.median(data))
        print("90 percent:", np.percentile(data, 90))
        print("95 percentile: ", np.percentile(data, 95))

    def print_stats(self):
        self.print_running_stats(self.metric, self.metric_name)
