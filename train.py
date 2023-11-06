# Imports
import os
import numpy as np
import pandas as pd
import datasets
from huggingface_hub import notebook_login
# Use token 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
# notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer



def train():
    dataset_name = "yuanbiao/imdb-card-pred"
    dataset = load_dataset(dataset_name, split="train")

    # base_model_name = "meta-llama/Llama-2-7b-hf"
    base_model_name = "NousResearch/Llama-2-7b-hf"
    # base_model_name = "HuggingFaceH4/zephyr-7b-beta"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = "./results"


    batch_size = 10
    max_steps = (100000) // batch_size
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=max_steps,
        num_train_epochs=10
        # save_strategy="epoch",
        # num_train_epochs=3
    )
    training_args = training_args.set_save(strategy="steps", steps=100)

    max_seq_length = 512
    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    token='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'
)
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_output_dir)

if __name__ == "__main__":
    train()