# Code Reference: https://www.philschmid.de/instruction-tune-llama-2

import os
import argparse
import wandb
from easydict import EasyDict as edict

from datasets import load_dataset
from trl import SFTTrainer
from utils import (
    load_config,
    load_training_args,
    dump_config,
    populate_default_arguments_for_config,
    dump_test_config_from_training_config,
)
from models import load_model_from_config


def train(config: edict):
    dataset = load_dataset(config.io.dataset_prefix + config.io.mode, split="train")

    training_args = load_training_args(config)

    model_dict = load_model_from_config(config)
    model = model_dict["model"]
    peft_config = model_dict["peft_config"]
    tokenizer = model_dict["tokenizer"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.io.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    final_output_dir = os.path.join(config.io.run_output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the config YAML file."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = populate_default_arguments_for_config(config)

    wandb.init(name=config.io.run_name)
    dump_config(config, os.path.join(config.io.run_output_dir, "config_snapshot.yaml"))
    dump_test_config_from_training_config(config)
    train(config)
