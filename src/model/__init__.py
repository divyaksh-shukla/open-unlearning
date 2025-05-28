from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
import os
import torch
import logging

hf_home = os.getenv("HF_HOME", default=None)


logger = logging.getLogger(__name__)


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args["attn_implementation"] == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            torch_dtype=torch_dtype, **model_args, cache_dir=hf_home
        )
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_cfg.model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer

def get_model_with_replaced_heads(model_cfg: DictConfig, finetuned_cfg: DictConfig, replace_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    assert finetuned_cfg is not None and finetuned_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    finetuned_model_args = finetuned_cfg.model_args
    
    torch_dtype = get_dtype(model_args)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            torch_dtype=torch_dtype, **model_args, cache_dir=hf_home
        )
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            **finetuned_model_args, cache_dir=hf_home
        )
        # check if lm_head and embed are in the model
        if not any("lm_head" in name for name, _ in model.named_parameters()):
            raise ValueError(
                f"Model {model_args.pretrained_model_name_or_path} does not have lm_head."
            )
        if not any("embed" in name for name, _ in model.named_parameters()):
            raise ValueError(
                f"Model {model_args.pretrained_model_name_or_path} does not have embed."
            )
        # Replace the heads of the model with the unlearned model
        for name, param in model.named_parameters():
            if "lm_head" in name and replace_cfg.get("replace_lm_head", True):
                logger.info(f"Replacing {name}")
                dtype = param.dtype
                param.data = finetuned_model.lm_head.weight.data
                if dtype == torch.float16:
                    param.data = param.data.half()
                elif dtype == torch.bfloat16:
                    param.data = param.data.bfloat16()
            elif "embed_tokens" in name and replace_cfg.get("replace_embed", True):
                logger.info(f"Replacing {name}")
                dtype = param.dtype
                param.data = finetuned_model.model.embed_tokens.weight.data
                if dtype == torch.float16:
                    param.data = param.data.half()
                elif dtype == torch.bfloat16:
                    param.data = param.data.bfloat16()
        
        model.to("cuda")
        # Delete the unlearned model to free up memory
        del finetuned_model
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_cfg.model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer

def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer
