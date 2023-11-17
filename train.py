# Code Reference: https://www.philschmid.de/instruction-tune-llama-2

import os
import argparse
import wandb
from easydict import EasyDict as edict

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from utils import (
    load_config,
    dataset_name_from_dataset_type,
    run_name_from_config,
    dump_config,
)
from models import load_model_from_config


def train(config: edict, output_dir: str):
    dataset = load_dataset(
        dataset_name_from_dataset_type(config.io.dataset_type), split="train"
    )
    max_seq_length = 512
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        num_train_epochs=config.training.num_train_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
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

    model_dict = load_model_from_config(config.model)
    model = model_dict["model"]
    peft_config = model_dict["peft_config"]
    tokenizer = model_dict["tokenizer"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the config YAML file."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_name = run_name_from_config(config)
    wandb.init(name=run_name)
    output_dir = os.path.join(config.io.output_dir, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump_config(config, os.path.join(output_dir, "config_snapshot.yaml"))
    train(config, output_dir)
