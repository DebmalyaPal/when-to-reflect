"""Verify the PyTorch runtime and run one deterministic Qwen generation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from when_to_reflect.config import load_config
from when_to_reflect.device import mps_available, select_device
from when_to_reflect.model import load_causal_lm

PROMPT = "In one sentence, name the capital of France."


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    device = select_device()
    model_id = config["model_id"]
    max_new_tokens = config["max_new_tokens"]

    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {mps_available()}")

    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor operation result: {(tensor * 2).sum().item()}")

    tokenizer, model = load_causal_lm(model_id, device)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Prompt: {PROMPT}")
    print(f"Generated output: {generated_text}")


if __name__ == "__main__":
    main()
