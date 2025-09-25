#!/usr/bin/env python3
"""
Training resume script for Higgs Audio v2
Handles resuming interrupted training sessions
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path
from transformers import TrainingArguments

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return None
    
    # Look for checkpoint directories
    checkpoint_dirs = [d for d in output_path.iterdir() 
                      if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        logger.info(f"No checkpoints found in {output_dir}")
        return None
    
    # Sort by checkpoint number
    def get_checkpoint_number(checkpoint_dir):
        try:
            return int(checkpoint_dir.name.split("-")[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoint_dirs.sort(key=get_checkpoint_number, reverse=True)
    latest_checkpoint = checkpoint_dirs[0]
    
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)

def validate_checkpoint(checkpoint_path):
    """Validate that checkpoint contains necessary files"""
    checkpoint_path = Path(checkpoint_path)
    
    required_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "training_args.bin",
        "trainer_state.json"
    ]
    
    # Check for either pytorch_model.bin or model.safetensors
    model_file_exists = (
        (checkpoint_path / "pytorch_model.bin").exists() or
        (checkpoint_path / "model.safetensors").exists()
    )
    
    missing_files = []
    for file in required_files:
        if file == "pytorch_model.bin":
            if not model_file_exists:
                missing_files.append("pytorch_model.bin or model.safetensors")
        elif not (checkpoint_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Checkpoint validation: Missing files: {missing_files}")
        return False
    
    logger.info("✓ Checkpoint validation passed")
    return True

def load_training_state(checkpoint_path):
    """Load training state from checkpoint"""
    try:
        trainer_state_path = Path(checkpoint_path) / "trainer_state.json"
        
        if not trainer_state_path.exists():
            logger.error("trainer_state.json not found in checkpoint")
            return None
        
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        logger.info(f"Training state loaded:")
        logger.info(f"  Global step: {trainer_state.get('global_step', 'N/A')}")
        logger.info(f"  Epoch: {trainer_state.get('epoch', 'N/A')}")
        logger.info(f"  Best metric: {trainer_state.get('best_metric', 'N/A')}")
        
        return trainer_state
        
    except Exception as e:
        logger.error(f"Failed to load training state: {e}")
        return None

def resume_training_command(checkpoint_path, original_args):
    """Generate the command to resume training"""
    
    # Base command
    cmd_parts = [
        "python3 trainer/trainer.py",
        f"--resume_from_checkpoint {checkpoint_path}"
    ]
    
    # Add original arguments (excluding conflicting ones)
    skip_args = {"resume_from_checkpoint", "overwrite_output_dir"}
    
    for arg, value in original_args.items():
        if arg in skip_args:
            continue
        
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{arg}")
        else:
            cmd_parts.append(f"--{arg} {value}")
    
    return " \\\n  ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description="Resume Higgs Audio v2 training")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory to check for checkpoints")
    parser.add_argument("--model_path", type=str, 
                       default="/root/data/higgs/weights/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--audio_tokenizer_path", type=str, 
                       default="/root/data/higgs/weights/higgs-audio-v2-tokenizer")
    parser.add_argument("--train_data_dir", type=str, 
                       default="/root/data/higgs/balanced_tts_dataset_higgs_mini")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--execute", action="store_true", 
                       help="Execute the resume command instead of just printing it")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HIGGS AUDIO V2 TRAINING RESUME HELPER")
    logger.info("=" * 60)
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.output_dir)
    
    if not latest_checkpoint:
        logger.error("No valid checkpoint found. Cannot resume training.")
        logger.info("To start fresh training, use the regular training script.")
        sys.exit(1)
    
    # Validate checkpoint
    if not validate_checkpoint(latest_checkpoint):
        logger.error("Checkpoint validation failed. Cannot resume training.")
        sys.exit(1)
    
    # Load training state
    training_state = load_training_state(latest_checkpoint)
    
    if training_state:
        logger.info(f"Training can be resumed from step {training_state.get('global_step', 'N/A')}")
    
    # Generate resume command
    original_args = {
        "model_path": args.model_path,
        "audio_tokenizer_path": args.audio_tokenizer_path,
        "train_data_dir": args.train_data_dir,
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "use_lora": args.use_lora,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "logging_steps": 50,
        "save_steps": 200,
        "report_to": "tensorboard"
    }
    
    resume_cmd = resume_training_command(latest_checkpoint, original_args)
    
    logger.info("=" * 60)
    logger.info("RESUME COMMAND:")
    logger.info("=" * 60)
    print(resume_cmd)
    print()
    
    if args.execute:
        logger.info("Executing resume command...")
        import subprocess
        
        try:
            # Execute the command
            result = subprocess.run(resume_cmd, shell=True, check=True)
            logger.info("✓ Training resumed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Training resume failed with exit code {e.returncode}")
            sys.exit(1)
    else:
        logger.info("To execute this command, add --execute flag")
        logger.info("Or copy and run the command above manually")

if __name__ == "__main__":
    main()