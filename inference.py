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
            pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def inference(self, prompt):
        pass

class BinaryInference(CardinalityInference):
    """
    Run inference and decode with binary format.
    """
    def __init__(self, model, tokenizer, device="cuda:0"):
        super(BinaryInference, self).__init__(model, tokenizer, device="cuda:0")
    
    def inference(self, prompt):
        output_text = self.get_output_text(prompt)

        # A regex pattern that matches the binary format
        binary_pattern = re.compile(r"([0-1]+)")

        # Find the matching characters
        binary_str = binary_pattern.match(output_text).groups[0]
        

