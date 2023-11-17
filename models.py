import torch
from easydict import EasyDict
from lib import TOKEN
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


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


def default_quantization_config() -> BitsAndBytesConfig:
    # Quantization config in QLoRA paper
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def peft_model(
    model_name: str,
    torch_dtype: any = "auto",
    use_flash_attention2: bool = True,
    lora_config: LoraConfig = None,
    qconfig: BitsAndBytesConfig = None,
) -> dict:
    """
    Load LLM and tokenizer from Huggingface by model name.
    Will return a dictionary with keys "model", "tokenizer" and "peft_config".
    For LoRA and QLoRA models, the peft_config will be a LoraConfig object. Otherwise it will be none.
    For QLora the quantization config will be a BitsAndBytesConfig object.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconfig,
        use_flash_attention_2=use_flash_attention2,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        token=TOKEN,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()
    
    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return {"model": model, "tokenizer": tokenizer, "peft_config": lora_config}


def load_model_from_config(model_config: EasyDict):
    assert model_config.adapter in [
        "none",
        "lora",
        "qlora",
    ], "model adapter not supported. Must be one of the following: qlora, lora or none."

    quantization_config = (
        default_quantization_config() if model_config.adapter == "qlora" else None
    )

    torch_dtype = torch.bfloat16 if model_config.use_bf16 else "auto"

    lora_config = None if model_config.adapter == "none" else default_lora_config()

    return peft_model(
        model_name=model_config.model_name,
        torch_dtype=torch_dtype,
        use_flash_attention2=model_config.use_flash_attention2,
        lora_config=lora_config,
        qconfig=quantization_config,
    )
