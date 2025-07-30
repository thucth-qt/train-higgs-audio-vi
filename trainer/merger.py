#!/usr/bin/env python3
"""
LoRA Merger for Higgs Audio v2
Merges LoRA adapters back into the base model
"""

import os
import torch
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
import json
import datetime

try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    HIGGS_AVAILABLE = True
    logging.info("Successfully imported Higgs Audio specific modules.")
except ImportError:
    HIGGS_AVAILABLE = False
    from transformers import AutoModel 
    logging.warning(
        "Could not import Higgs Audio modules from 'boson_multimodal'. "
        "Falling back to `AutoModel`. This will likely fail if the model "
        "type 'higgs_audio' is not registered with Transformers."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiggsAudioLoRaMerger:
    """Merger for LoRA adapters with Higgs Audio v2 models"""
    
    def __init__(self, base_model_path: str, lora_adapter_path: str):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
    def load_base_model(self):
        """Load the base Higgs Audio model"""
        logger.info(f"Loading base model from {self.base_model_path}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        # 确保 tokenizer 有一个 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer's pad_token was None, set to eos_token: {self.tokenizer.pad_token}")
        
        # 修改加载模型的方式，不再使用通用的 AutoModel，
        # 而是使用从 boson_multimodal 导入的特定 HiggsAudioModel 类。
        if HIGGS_AVAILABLE:
            logger.info("Using specific HiggsAudioModel and HiggsAudioConfig to load the base model.")
            config = HiggsAudioConfig.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            self.base_model = HiggsAudioModel.from_pretrained(
                self.base_model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            logger.error("The `boson_multimodal` library is required to load the 'higgs_audio' model type but it was not found.")
            raise ImportError("Please install the required 'boson_multimodal' library to proceed.")
        
        logger.info("Base model loaded successfully")
        return self.base_model
        
    def load_lora_model(self):
        """Load model with LoRA adapters"""
        logger.info(f"Loading LoRA adapters from {self.lora_adapter_path}")
        
        # Load LoRA config
        peft_config = PeftConfig.from_pretrained(self.lora_adapter_path)
        logger.info(f"LoRA config: {peft_config}")
        
        # Load model with LoRA adapters
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_adapter_path,
            config=peft_config
        )
        
        logger.info("LoRA model loaded successfully")
        return self.lora_model
        
    def merge_and_save(self, output_path: str, save_tokenizer: bool = True):
        """Merge LoRA weights and save the merged model"""
        logger.info("Merging LoRA weights into base model...")
        
        # Merge LoRA weights
        merged_model = self.lora_model.merge_and_unload()
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(
            output_path,
            # safe_serialization=True # 某些自定义模型可能不支持，如果保存失败可以注释掉
        )
        
        # Save tokenizer
        if save_tokenizer:
            logger.info(f"Saving tokenizer to {output_path}")
            self.tokenizer.save_pretrained(output_path)
            
        # Save model info
        model_info = {
            "base_model": self.base_model_path,
            "lora_adapter": self.lora_adapter_path,
            "merged_at": datetime.datetime.utcnow().isoformat(),
            "model_type": "higgs-audio-v2-lora-merged"
        }
        
        with open(output_dir / "merge_info.json", 'w') as f:
            json.dump(model_info, f, indent=4)
            
        logger.info(f"Model merged and saved successfully to {output_path}")
        
    def compare_models(self, test_input: str = None):
        """Compare outputs between base model and LoRA model"""
        if test_input is None:
            test_input = "The sun rises in the east and sets in the west."
            
        logger.info(f"Comparing base model and LoRA model outputs using input: '{test_input}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.base_model.device) # 确保输入在模型所在的设备上
        
        # Get base model output
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)
            
        # Get LoRA model output  
        with torch.no_grad():
            lora_outputs = self.lora_model(**inputs)
            
        # Compare outputs
        diff = None
        if hasattr(base_outputs, 'last_hidden_state') and hasattr(lora_outputs, 'last_hidden_state'):
            base_tensor = base_outputs.last_hidden_state
            lora_tensor = lora_outputs.last_hidden_state
            tensor_name = "last_hidden_state"
        elif hasattr(base_outputs, 'logits') and hasattr(lora_outputs, 'logits'):
            base_tensor = base_outputs.logits
            lora_tensor = lora_outputs.logits
            tensor_name = "logits"
        else:
            logger.warning("Could not find 'last_hidden_state' or 'logits' in model outputs to compare.")
            return {}

        diff = torch.abs(base_tensor - lora_tensor).mean().item()
        logger.info(f"Average absolute difference in '{tensor_name}': {diff:.6f}")
        
        base_mean = torch.abs(base_tensor).mean().item()
        if base_mean > 1e-9:
            rel_change = (diff / base_mean) * 100
            logger.info(f"Relative change: {rel_change:.2f}%")
        else:
            logger.info("Base tensor mean is close to zero, relative change is not meaningful.")

        return {
            "base_output": base_outputs,
            "lora_output": lora_outputs,
            "difference": diff
        }

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with Higgs Audio v2 base model")
    
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True,
        help="Path to the local base Higgs Audio model directory"
    )
    parser.add_argument(
        "--lora_adapter_path", 
        type=str, 
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--compare_models", 
        action="store_true",
        help="Compare base and LoRA models before merging"
    )
    parser.add_argument(
        "--test_input", 
        type=str,
        default="The sun rises in the east and sets in the west.",
        help="Test input for model comparison"
    )
    parser.add_argument(
        '--no_save_tokenizer', 
        action='store_false', 
        dest='save_tokenizer',
        help="Do not save the tokenizer with the merged model"
    )
    parser.set_defaults(save_tokenizer=True)
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.base_model_path):
        raise ValueError(f"Base model path does not exist: {args.base_model_path}")
    if not os.path.exists(args.lora_adapter_path):
        raise ValueError(f"LoRA adapter path does not exist: {args.lora_adapter_path}")
        
    # Initialize merger
    merger = HiggsAudioLoRaMerger(args.base_model_path, args.lora_adapter_path)
    
    # Load models
    merger.load_base_model() # This now handles pad_token
    merger.load_lora_model()
    
    # Compare models if requested
    if args.compare_models:
        merger.compare_models(args.test_input)
        logger.info("Model comparison completed")
        
    # Merge and save
    merger.merge_and_save(args.output_path, args.save_tokenizer)
    
    logger.info("LoRA merge process completed successfully!")

if __name__ == "__main__":
    main()
