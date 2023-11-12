# Code Reference: https://www.philschmid.de/instruction-tune-llama-2

import os
import argparse
import wandb

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from utils import load_config
from models import load_llm_from_huggingface, lora_wrapper, qlora_wrapper

def train(config):
    dataset = load_dataset(config.dataset_name, split="train")
    max_seq_length = 512
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=True,
        num_train_epochs=config.num_train_epochs,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )
    training_args = training_args.set_dataloader(train_batch_size=config.batch_size)

    model_dict = load_llm_from_huggingface(config.model_name)
    if config.adapter == "lora":
        model_dict = lora_wrapper(model_dict, config.lora_config)
    elif config.adapter == "qlora":
        model_dict = qlora_wrapper(model_dict, config.lora_config)

    model = model_dict["model"]
    peft_config = model_dict["lora_config"]
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

    final_output_dir = os.path.join(config.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    config = load_config(args.config)
    wandb.init(name=config.run_name)
    train(args.dataset, args.checkpoint_dir, args.batchSize)
