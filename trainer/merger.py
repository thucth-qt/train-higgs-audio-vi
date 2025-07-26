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
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import json

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
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
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
            save_function=torch.save,
            safe_serialization=True
        )
        
        # Save tokenizer
        if save_tokenizer:
            logger.info("Saving tokenizer...")
            self.tokenizer.save_pretrained(output_path)
            
        # Save model info
        model_info = {
            "base_model": self.base_model_path,
            "lora_adapter": self.lora_adapter_path,
            "merged_at": str(torch.utils.data.dataset.T),
            "model_type": "higgs-audio-v2-lora-merged"
        }
        
        with open(output_dir / "merge_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
            
        logger.info(f"Model merged and saved successfully to {output_path}")
        
    def compare_models(self, test_input: str = None):
        """Compare outputs between base model and LoRA model"""
        if test_input is None:
            test_input = "The sun rises in the east and sets in the west."
            
        logger.info("Comparing base model and LoRA model outputs...")
        
        # Tokenize input
        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Get base model output
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)
            
        # Get LoRA model output  
        with torch.no_grad():
            lora_outputs = self.lora_model(**inputs)
            
        # Compare outputs
        if hasattr(base_outputs, 'last_hidden_state') and hasattr(lora_outputs, 'last_hidden_state'):
            base_hidden = base_outputs.last_hidden_state
            lora_hidden = lora_outputs.last_hidden_state
            
            # Calculate difference
            diff = torch.abs(base_hidden - lora_hidden).mean().item()
            logger.info(f"Average absolute difference in hidden states: {diff:.6f}")
            
            # Calculate relative change
            rel_change = (diff / torch.abs(base_hidden).mean().item()) * 100
            logger.info(f"Relative change: {rel_change:.2f}%")
            
        return {
            "base_output": base_outputs,
            "lora_output": lora_outputs,
            "difference": diff if 'diff' in locals() else None
        }

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with Higgs Audio v2 base model")
    
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to base Higgs Audio model"
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
        help="Path to save merged model"
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
        "--save_tokenizer", 
        action="store_true", 
        default=True,
        help="Save tokenizer with merged model"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.lora_adapter_path):
        raise ValueError(f"LoRA adapter path does not exist: {args.lora_adapter_path}")
        
    # Initialize merger
    merger = HiggsAudioLoRaMerger(args.base_model_path, args.lora_adapter_path)
    
    # Load models
    base_model = merger.load_base_model()
    lora_model = merger.load_lora_model()
    
    # Compare models if requested
    if args.compare_models:
        comparison_results = merger.compare_models(args.test_input)
        logger.info("Model comparison completed")
        
    # Merge and save
    merger.merge_and_save(args.output_path, args.save_tokenizer)
    
    logger.info("LoRA merge process completed successfully!")

if __name__ == "__main__":
    main()