"""Hugging Face causal-language-model loading."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_id: str, device: torch.device):
    """Load a causal language model and tokenizer for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model
