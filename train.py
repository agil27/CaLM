# Imports
"""
From this link: https://www.philschmid.de/instruction-tune-llama-2
"""
import os
import numpy as np
import pandas as pd
import datasets
import argparse
from huggingface_hub import notebook_login
# Use token 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
# notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft import LoraConfig
from trl import SFTTrainer


def load_model_and_tokenizer():
    """
    Load model and tokenizer
    """
    base_model_name = "NousResearch/Llama-2-7b-hf"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        use_flash_attention_2=True,
        device_map="auto",
        trust_remote_code=True,
        token='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'
        )
    base_model.config.use_cache = False
    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model, tokenizer

def train(output_dir):
    """
    Load dataset
    """
    dataset_name = "yuanbiao/imdb-card-pred"
    dataset = load_dataset(dataset_name, split="train")

    max_seq_length = 512
    """
    Prepare model for training
    """
    model, tokenizer = load_model_and_tokenizer()
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)


    batch_size = 100
    max_steps = (100000) // batch_size
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03
    )

    training_args = training_args.set_save(strategy="steps", steps=100)
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
    trainer.model.save_pretrained(final_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script with two string parameters')
    parser.add_argument('checkpoint_dir', type=str, help='Directory for checkpoints')
    args = parser.parse_args()
    train(args.checkpoint_dir)
