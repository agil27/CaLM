import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft import LoraConfig
from lib import TOKEN


def default_lora_config() -> LoraConfig:
    # LoRA config in QLoRA paper
    return LoraConfig(
        inference_mode=False,
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_llm_from_huggingface(model_name: str) -> dict:
    """
    Load LLM and tokenizer from Huggingface by model name.
    Will return a dictionary with keys "model" and "tokenizer".
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=TOKEN,
    )

    model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return {"model": model, "tokenizer": tokenizer}


def lora_wrapper(base_model_dict: dict, lora_config: LoraConfig) -> dict:
    """
    Wrap the model with LoRA adapter to make it partially trainable.
    Will return a dictionary with keys "model", "tokenizer" and "lora_config".
    """
    return {
        "model": get_peft_model(base_model_dict["model"], lora_config),
        "tokenizer": base_model_dict["tokenizer"],
        "lora_config": lora_config,
    }


def qlora_wrapper(base_model_dict: dict, lora_config: LoraConfig):
    """
    Wrap the model with QLoRA adapter to make it partially trainable.
    QLora will use a 4-bit quantization to reduce training overheads.
    Will return a dictionary with keys "model", "tokenizer" and "lora_config".
    """
    return {
        "model": get_peft_model(
            prepare_model_for_kbit_training(base_model_dict["model"]), lora_config
        ),
        "tokenizer": base_model_dict["tokenizer"],
        "lora_config": lora_config,
    }
