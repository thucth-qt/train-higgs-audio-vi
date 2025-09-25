#!/usr/bin/env python3
"""
Script to convert a full model checkpoint containing LoRA weights
into a standalone LoRA adapter directory.

This script is designed to be used with checkpoints generated from the
Higgs Audio v2 training script (trainer.py).
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from typing import Dict
from safetensors.torch import load_file

# Import PEFT and Transformers libraries
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoConfig

# Try to import Higgs Audio related modules
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False

class HiggsAudioModelWrapper(nn.Module):
    """
    Simplified wrapper to load the Higgs Audio model structure.
    This is a stripped-down version of the wrapper in trainer.py,
    focused only on loading the model for conversion.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        super().__init__()
        if not HIGGS_AVAILABLE:
            raise ImportError("Higgs Audio modules not available. This script requires them.")

        print(f"[INFO] Loading base model structure from {model_path}...")
        self.model = HiggsAudioModel.from_pretrained(
            config=HiggsAudioConfig.from_pretrained(model_path),
            pretrained_model_name_or_path=model_path,
            device_map=device,
        )
        self.config = self.model.config
        self.model = self.model.to(device)

    @property
    def device(self):
        return self.model.device

def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """
    Applies LoRA configuration to the model, making it a PeftModel.
    This function is copied from trainer.py to ensure consistency.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # Set to False to allow loading weights
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        auto_mapping=True
    )

    model = model.to(device)

    # Apply LoRA to the text_model component, as done in training
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)

    model = model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Higgs Audio training checkpoint to a standalone LoRA adapter.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base Higgs Audio model directory (the one used to start training).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the training checkpoint directory (e.g., ./runs/output/checkpoint-9000).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the extracted LoRA adapter (e.g., ./runs/output/lora_adapters).")

    # --- LoRA Configuration ---
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+',
                        default=["q_proj", "v_proj", "k_proj", "o_proj"],
                        help="Modules to target with LoRA. Should match the training configuration.")

    args = parser.parse_args()

    if not HIGGS_AVAILABLE:
        print("Error: Higgs Audio modules ('boson_multimodal') are not installed. Please install them to proceed.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load the base model wrapper
    model_wrapper = HiggsAudioModelWrapper(args.base_model_path, device=device)
    print("Base model loaded successfully.")

    # 2. Apply LoRA configuration to create the PeftModel structure
    print("Applying LoRA configuration...")
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": args.lora_target_modules
    }
    model_wrapper = setup_lora_config(model_wrapper, lora_config)
    print("LoRA configuration applied.")

    # 3. Load the state dictionary from the checkpoint
    print(f"Loading weights from checkpoint: {args.checkpoint_path}...")
    state_dict = {}
    
    # Check for various model file formats
    index_path = os.path.join(args.checkpoint_path, "pytorch_model.bin.index.json")
    single_file_path = os.path.join(args.checkpoint_path, "pytorch_model.bin")
    safetensors_path = os.path.join(args.checkpoint_path, "model.safetensors")

    if os.path.exists(index_path):
        print("Sharded checkpoint detected. Consolidating weights...")
        with open(index_path, "r") as f:
            index = json.load(f)
        
        # Collect all shard files from the index
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            shard_path = os.path.join(args.checkpoint_path, shard_file)
            if os.path.exists(shard_path):
                shard_state_dict = torch.load(shard_path, map_location='cpu')
                state_dict.update(shard_state_dict)
            else:
                print(f"Warning: Shard file not found: {shard_path}")

    elif os.path.exists(single_file_path):
        print("Single-file checkpoint detected.")
        state_dict = torch.load(single_file_path, map_location='cpu')
        
    elif os.path.exists(safetensors_path):
        print("Safetensors checkpoint detected.")
        state_dict = load_file(safetensors_path, device="cpu")
        
    else:
        print(f"Error: Could not find 'pytorch_model.bin', 'pytorch_model.bin.index.json', or 'model.safetensors' in {args.checkpoint_path}")
        return

    # Load the consolidated state dictionary into the model
    model_wrapper.load_state_dict(state_dict)
    print("Checkpoint weights loaded successfully into model.")

    # 4. Find the PeftModel part and save the adapters
    print(f"Saving LoRA adapters to {args.output_path}...")

    peft_model_part = None
    if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'text_model') and isinstance(model_wrapper.model.text_model, PeftModel):
        peft_model_part = model_wrapper.model.text_model
    elif hasattr(model_wrapper, 'model') and isinstance(model_wrapper.model, PeftModel):
        peft_model_part = model_wrapper.model
    elif isinstance(model_wrapper, PeftModel):
        peft_model_part = model_wrapper

    if peft_model_part:
        peft_model_part.save_pretrained(args.output_path)
        print(f"Successfully saved LoRA adapters to {args.output_path}")
        print("Conversion complete.")
    else:
        print("Error: Could not find a PeftModel instance in the model hierarchy. Ensure LoRA was used during training.")

if __name__ == "__main__":
    main()
