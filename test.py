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

def calc_qerror(estimated_cardinality, actual_cardinality):
    return max(estimated_cardinality / actual_cardinality, actual_cardinality / estimated_cardinality)

def print_qerror_stats(qerror_list):
    qerror_mean = np.mean(qerror_list)
    print("\tmean", qerror_mean)
    qerror_median = np.median(qerror_list)
    print("\tmedian", qerror_median)
    qerror_max = np.max(qerror_list)
    print("\tmax", qerror_max)

    # Computing the 90th percentile
    percentile_90 = np.percentile(qerror_list, 90)
    print("\t90th percentile:", percentile_90)

    # Computing the 95th percentile
    percentile_95 = np.percentile(qerror_list, 95)
    print("\t95th percentile:", percentile_95)
    pass

def extract_cardinality(text):
    """
    This helper function is to extract the output cardinality from LLM output
    """
    i = 0
    while i < len(text) and (not (text[i] >= '0' and text[i] <= '9')):
        i += 1
    
    cardinality = 0
    while i < len(text):
        if text[i] >= '0' and text[i] <= '9':
            cardinality = 10 * cardinality + (ord(text[i]) - ord('0'))
        else:
            break
        i += 1
    return cardinality

def test(checkpoint_dir, device="cuda:0"):
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir, 
                                                     device_map=device, 
                                                     torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dataset_name = "yuanbiao/imdb-card-pred"
    dataset = load_dataset(dataset_name, split="train")
    qerror_list = []
    step = 1
    for data in dataset:
        text = data['text']
        tokens = text.split(' ')
        gt_cardinality = int(tokens[-1])

        # Process prompts
        prompt = " ".join(tokens[:-1]) + " "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id)


        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_cardinality = extract_cardinality(output_text[len(prompt): ])

        print(f"\ngt_cardinality: {gt_cardinality}, output_cardinality: {output_cardinality}")
        print(f"\tgt_text: {text}") 
        print(f"\toutput_text: {output_text}") 
        # Calculate qerror
        qerror = calc_qerror(output_cardinality, gt_cardinality)
        qerror_list.append(qerror)

        step += 1
        if step % 10 == 0:
            print(f"{step} / {len(data)}")
            print_qerror_stats(qerror_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script with two string parameters')
    parser.add_argument('checkpoint_dir', type=str, help='Directory for checkpoints')
    parser.add_argument('-gpu_device', type=str, help='GPU device', default="cuda:0")

    args = parser.parse_args()
    test(args.checkpoint_dir, args.gpu_device)
