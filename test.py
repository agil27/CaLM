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

def extract_cardinality(text, mode='decimal'):
    """
    This helper function is to extract the output cardinality from LLM output
    """
    if mode == 'decimal' or 'binary':
        # Calculate the cardinality when using decimal or binary represetnation
        upper_char = '9' if mode == 'decimal' else '1'
        base = 10 if mode == 'decimal' else 2
        i = 0
        # Skip characters that are not digits
        while i < len(text) and (not (text[i] >= '0' and text[i] <= upper_char)):
            i += 1
    
        cardinality = 0
        # Start to calculate the cardinality
        while i < len(text):
            if text[i] >= '0' and text[i] <= upper_char :
                cardinality = base * cardinality + (ord(text[i]) - ord('0'))
            else:
                break
            i += 1
        return cardinality
    elif mode == 'science':
        pass
    else:
        raise RuntimeError("Unsupported represetation for cardinality")


def test(checkpoint_dir, dataset_name, mode, device="cuda:0"):
    # Load model from the checkpoint
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir, 
                                                     device_map=device, 
                                                     torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")
    qerror_list = []

    step = 1
    for data in dataset:
        text = data['text']
        tokens = text.split(' ')
        gt_cardinality = extract_cardinality(tokens[-1], mode)

        # Process prompts
        prompt = " ".join(tokens[:-1]) + " "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id)


        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_cardinality = extract_cardinality(output_text[len(prompt): ], mode)
        print(f"\nTest sample {step}/{len(dataset)}")
        print(f"\tgt_cardinality: {gt_cardinality}, output_cardinality: {output_cardinality}")
        print(f"\tgt_text: {text}") 
        print(f"\toutput_text: {output_text}") 
        # Calculate qerror
        qerror = calc_qerror(output_cardinality, gt_cardinality)
        qerror_list.append(qerror)

        step += 1
        if step % 10 == 0:
            print(f"\nFor the first {step} samples, here is the qerror stats")
            print_qerror_stats(qerror_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script with two string parameters')
    parser.add_argument('--checkpoint', type=str, help='Directory for checkpoints', default='result')
    parser.add_argument('--dataset', type=str, help='dataset', default="vic0428/imdb-card-pred-decimal")
    parser.add_argument('--mode', type=str, help='cardinality represetation', default="binary")
    parser.add_argument('--device', type=str, help='GPU device', default="cuda:0")

    args = parser.parse_args()
    # Start the testing 
    test(args.checkpoint,
         args.dataset,
         args.mode,
         args.device)
