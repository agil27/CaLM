import re
from transformers import Pipeline
import numpy as np
import torch


MAX_QERROR = 1e7


def calc_qerror(estimated_cardinality, actual_cardinality):
    if estimated_cardinality == 0 or actual_cardinality == 0:
        return MAX_QERROR
    return max(
        estimated_cardinality / actual_cardinality,
        actual_cardinality / estimated_cardinality,
    )


def batch_calc_qerror(estimated_cardinalities, actual_cardinalities):
    e_array = np.array(estimated_cardinalities)
    a_array = np.array(actual_cardinalities)
    e_over_a = np.divide(
        e_array, a_array, where=(a_array != 0), out=np.ones_like(e_array) * MAX_QERROR
    )
    a_over_e = np.divide(
        a_array, e_array, where=(e_array != 0), out=np.ones_like(e_array) * MAX_QERROR
    )
    return np.maximum(e_over_a, a_over_e)


def binary_decode(output_text):
    # A regex pattern that matches the binary format
    binary_pattern = re.compile(r"([0-1]+)")

    # Find the matching characters
    binary_strs = re.findall(binary_pattern, output_text)
    if len(binary_strs) == 0:
        return 0
    return int(binary_strs[0], 2)


def decimal_decode(output_text):
    decimal_pattern = re.compile(r"(\d+)")
    decimal_strs = re.findall(decimal_pattern, output_text)
    if len(decimal_strs) == 0:
        return 0
    return int(decimal_strs[0])


def science_decode(output_text):
    science_pattern = re.compile(r"(\d\.\d+e\d+)")
    science_strs = re.findall(science_pattern, output_text)
    if len(science_strs) == 0:
        return 0
    return int(float(science_strs[0]))


def decode_cardinality_and_calc_qerror(
    output_text: str, true_cardinality: int, decode_mode: str = "decimal"
) -> dict:
    estimated_cardinality = None
    if decode_mode == "binary":
        estimated_cardinality = binary_decode(output_text)
    elif decode_mode == "decimal":
        estimated_cardinality = decimal_decode(output_text)
    elif decode_mode == "science":
        estimated_cardinality = science_decode(output_text)
    else:
        raise ValueError(
            "Invalid decode mode. Should be one of decimal, binary or scientific."
        )
    qerror = calc_qerror(estimated_cardinality, true_cardinality)
    return {
        "estimated_cardinality": estimated_cardinality,
        "true_cardinality": true_cardinality,
        "qerror": qerror,
    }


def batch_decode_cardinality_and_calc_qerror(
    output_texts, true_cardinalities, decode_mode: str = "decimal"
) -> dict:
    estimated_cardinalities = []
    true_cardinalities = torch.Tensor(true_cardinalities).cpu().numpy()
    for i in range(len(output_texts)):
        if decode_mode == "binary":
            estimated_cardinalities.append(binary_decode(output_texts[i]))
        elif decode_mode == "decimal":
            estimated_cardinalities.append(decimal_decode(output_texts[i]))
        elif decode_mode == "science":
            estimated_cardinalities.append(science_decode(output_texts[i]))
        else:
            raise ValueError(
                "Invalid decode mode. Should be one of decimal, binary or scientific."
            )
    print(type(estimated_cardinalities), type(true_cardinalities))
    qerrors = batch_calc_qerror(estimated_cardinalities, true_cardinalities)
    return {
        "estimated_cardinality": np.array(estimated_cardinalities),
        "true_cardinality": np.array(true_cardinalities),
        "qerror": np.array(qerrors),
    }


class DecodeCardinalityPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        post_process_kwargs = {}
        if "max_length" in kwargs:
            forward_kwargs["max_length"] = kwargs["max_length"]
        if "decode_mode" in kwargs:
            post_process_kwargs["decode_mode"] = kwargs["decode_mode"]
        return preprocess_kwargs, forward_kwargs, post_process_kwargs

    def preprocess(self, raw_inputs, second_text=None):
        model_inputs = self.tokenizer(
            raw_inputs["prompt"], return_tensors=self.framework, padding="longest"
        )
        model_inputs["true_cardinality"] = raw_inputs["true_cardinality"]
        model_inputs["prompt"] = raw_inputs["prompt"]
        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        max_length = generate_kwargs.pop("max_length", 50)
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_outputs = {
            "output_text": output_text[len(model_inputs["prompt"]) :],
            "true_cardinality": model_inputs["true_cardinality"],
        }
        return model_outputs

    def postprocess(self, model_outputs, **generate_kwargs):
        decode_mode = generate_kwargs.pop("decode_mode", "decimal")
        return decode_cardinality_and_calc_qerror(
            model_outputs["output_text"], model_outputs["true_cardinality"], decode_mode
        )
