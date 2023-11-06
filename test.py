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
from peft import AutoPeftModelForCausalLM
import argparse

def test(checkpoint_dir, device="cuda:0"):
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir, 
                                                     device_map=device, 
                                                     torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dataset_name = "yuanbiao/imdb-card-pred"
    dataset = load_dataset(dataset_name, split="train")
    for data in dataset:
        text = data['text']
        tokens = text.split(' ')
        gt_cardinality = int(tokens[-1])

        # Process prompts
        prompt = " ".join(tokens[:-1])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id)

        print(outputs)
        exit(0)
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script with two string parameters')
    parser.add_argument('checkpoint_dir', type=str, help='Directory for checkpoints')
    parser.add_argument('gpu_device', type=str, help='GPU device')

    args = parser.parse_args()
    test(args.checkpoint_dir, args.gpu_device)
