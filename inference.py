import torch
import re


def qerror(estimated_cardinality, actual_cardinality):
    return max(
        estimated_cardinality / actual_cardinality,
        actual_cardinality / estimated_cardinality,
    )


class CardinalityInference:
    """
    A Virtual Inference class for huggingface LLM cardinality prediction.
    model: A huggingface model.
    tokenizer: A huggingface tokenizer
    """

    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else "cpu"

    def get_output_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

    def inference(self, prompt):
        pass


class BinaryInference(CardinalityInference):
    """
    Run inference and decode with binary format.
    """

    def __init__(self, model, tokenizer, device="cuda:0"):
        super(BinaryInference, self).__init__(model, tokenizer, device)

    def decode(self, output_text):
        # A regex pattern that matches the binary format
        binary_pattern = re.compile(r"([0-1]+)")

        # Find the matching characters
        binary_str = re.findall(binary_pattern, output_text)[0]
        return int(binary_str, 2)
    
    def inference(self, prompt):
        return self.decode(self.get_output_text(prompt))


class DecimalInference(CardinalityInference):
    """
    Run inference and decode with decimal format
    """
    def __init__(self, model, tokenizer, device="cuda:0"):
        super(DecimalInference, self).__init__(model, tokenizer, device)

    def decode(self, output_text):
        decimal_pattern = re.compile(r"(\d+)")
        decimal_str = re.findall(decimal_pattern, output_text)[0]
        return int(decimal_str)

    def inference(self, prompt):
        return self.decode(self.get_output_text(prompt))


class ScientificInference(CardinalityInference):
    """
    Run inference and decode with scientific format
    """
    def __init__(self, model, tokenizer, device="cuda:0"):
        super(ScientificInference, self).__init__(model, tokenizer, device)

    def decode(self, output_text):
        scientific_pattern = re.compile(r"(\d\.\d+e\d+)")
        scientific_str = re.findall(scientific_pattern, output_text)[0]
        return int(float(scientific_str))
    
    def inference(self, prompt):
        return self.decode(self.get_output_text(prompt))

        