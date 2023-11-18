import re
from transformers import Pipeline


def calc_qerror(estimated_cardinality, actual_cardinality):
    return max(
        estimated_cardinality / actual_cardinality,
        actual_cardinality / estimated_cardinality,
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


def science_decode(output_text):
    science_pattern = re.compile(r"(\d\.\d+e\d+)")
    science_str = re.findall(science_pattern, output_text)[0]
    return int(float(science_str))


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
        estimated_cardinality = None
        output_text = model_outputs["output_text"]
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

        true_cardinality = model_outputs["true_cardinality"]
        qerror = calc_qerror(estimated_cardinality, true_cardinality)
        return {
            "estimated_cardinality": estimated_cardinality,
            "true_cardinality": true_cardinality,
            "qerror": qerror,
        }
