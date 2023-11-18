import torch
import re
from transformers import Pipeline
from abc import ABC, abstractmethod


def qerror(estimated_cardinality, actual_cardinality):
    return max(
        estimated_cardinality / actual_cardinality,
        actual_cardinality / estimated_cardinality,
    )


def DecodeCardinalityPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        post_process_kwargs = {}
        if "max_length" in kwargs:
            forward_kwargs["max_length"] = kwargs["max_length"]
        if "decode_mode" in kwargs:
            post_process_kwargs["decode_mode"] = kwargs["decode_mode"]
        return preprocess_kwargs, forward_kwargs, post_process_kwargs

    def preprocess(self, text, second_text=None):
        inputs = self.tokenizer(text, return_tensors=self.framework)
        inputs["prompt"] = text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        max_length = generate_kwargs.pop("max_length", 512)
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text[len(model_inputs["prompt"]) :]

    def postprocess(self, model_outputs, **generate_kwargs):
        decode_mode = generate_kwargs.pop("decode_mode", "decimal")
        if decode_mode == "binary":
            return binary_decode(model_outputs)
        elif decode_mode == "decimal":
            return decimal_decode(model_outputs)
        elif decode_mode == "scientific":
            return scientific_decode(model_outputs)
        else:
            raise ValueError(
                "Invalid decode mode. Should be one of decimal, binary or scientific."
            )
            

def binary_decode(output_text):
    # A regex pattern that matches the binary format
    binary_pattern = re.compile(r"([0-1]+)")

    # Find the matching characters
    binary_str = re.findall(binary_pattern, output_text)[0]
    return int(binary_str, 2)


def decimal_decode(output_text):
    decimal_pattern = re.compile(r"(\d+)")
    decimal_str = re.findall(decimal_pattern, output_text)[0]
    return int(decimal_str)


def scientific_decode(output_text):
    scientific_pattern = re.compile(r"(\d\.\d+e\d+)")
    scientific_str = re.findall(scientific_pattern, output_text)[0]
    return int(float(scientific_str))
