import os
from typing import Dict, Optional, Union, Any
import torch
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def setup_tokenizer(model_name: str, config: Dict[str, Any]) -> AutoTokenizer:
    """
    Initialize and configure tokenizer with proper settings.
    
    Args:
        model_name: Name or path of the model to load tokenizer for
        config: Configuration dictionary containing tokenizer settings
        
    Returns:
        Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=config.get("hf_token")
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = config["tokenizer"].get("pad_token_id", tokenizer.eos_token_id)
    
    if config["tokenizer"].get("chat_template"):
        tokenizer.chat_template = config["tokenizer"]["chat_template"]
    
    return tokenizer

def get_model_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model initialization kwargs based on config.
    
    Args:
        config: Configuration dictionary containing model settings
        
    Returns:
        Dictionary of kwargs for model initialization
    """
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    model_kwargs = {
        "token": config.get("hf_token"),
        "torch_dtype": quant_storage_dtype,
    }

    if not config.get("use_accelerator", False):
        model_kwargs["device_map"] = "auto"

    # Add quantization config if needed
    if config.get("quantization", {}).get("enabled", False):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    # Add flash attention if enabled
    if config["model_config"].get("use_flash_attention", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    return model_kwargs

def load_base_model(
    model_name: str,
    config: Dict[str, Any],
    prepare_for_kbit: bool = False,
    cache_dir=None,
    save_base_model_to_cache=False
) -> AutoModelForCausalLM:
    """
    Load base model with proper configuration.
    
    Args:
        model_name: Name or path of the model to load
        config: Configuration dictionary
        prepare_for_kbit: Whether to prepare model for k-bit training
        
    Returns:
        Loaded model instance
    """
    model_kwargs = get_model_kwargs(config)
    if cache_dir is not None and os.path.exists(cache_dir):
        model_name=cache_dir
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if save_base_model_to_cache and  not os.path.exists(cache_dir):
        model.save_pretrained(cache_dir)

    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(model)
    
    return model

def configure_lora(
    model: AutoModelForCausalLM,
    config: Dict[str, Any]
) -> Optional[PeftModel]:
    """
    Configure LoRA for the model if enabled in config.
    
    Args:
        model: Base model to configure LoRA for
        config: Configuration dictionary containing LoRA settings
        
    Returns:
        Model with LoRA configuration if enabled, None otherwise
    """
    if not config["lora"].get("enable_training", False):
        return None
        
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    
    return get_peft_model(model, lora_config)

def load_adapter(
    model: AutoModelForCausalLM,
    adapter_path: str
) -> PeftModel:
    """
    Load a pre-trained adapter for the model.
    
    Args:
        model: Base model to load adapter for
        adapter_path: Path to the adapter
        
    Returns:
        Model with loaded adapter
    """
    model = PeftModel.from_pretrained(model, adapter_path)

    # Ensure all LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            
    return model

def setup_model(
    model_name: str,
    config: Dict[str, Any],
    model_type: str = "student"
) -> Union[AutoModelForCausalLM, PeftModel]:
    """
    Main function to setup a model with all configurations.
    
    Args:
        model_name: Name or path of the model to load
        config: Configuration dictionary
        model_type: Type of model ("student" or "teacher")
        
    Returns:
        Configured model instance
    """
    prepare_for_kbit = model_type == "student" and config["lora"].get("enable_training", False)
    model = load_base_model(model_name, config, prepare_for_kbit)
    
    # Handle adapter loading
    adapter_path = config["models"].get(f"{model_type}_adapter")
    if adapter_path:
        return load_adapter(model, adapter_path)
    
    # Configure LoRA for student model
    if model_type == "student" and config["lora"].get("enable_training", False):
        return configure_lora(model, config)
    
    return model

def load_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all required models based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing loaded models and tokenizers
    """
    result = {}
    
    # Load student model and tokenizer
    result["student_tokenizer"] = setup_tokenizer(
        config["models"]["student"],
        config
    )
    
    result["student_model"] = setup_model(
        config["models"]["student"],
        config,
        "student"
    )
    
    # Load teacher model if needed
    if config["models"].get("teacher") and not config["dataset"].get("logits_file"):
        result["teacher_model"] = setup_model(
            config["models"]["teacher"],
            config,
            "teacher"
        )
    
    return result
