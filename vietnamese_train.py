#!/usr/bin/env python3
"""
Generated Vietnamese TTS Training Script
Run this script to train Higgs Audio v2 for Vietnamese TTS
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the existing trainer
from trainer.trainer import main as train_main
import argparse

if __name__ == "__main__":
    # Override sys.argv with Vietnamese-specific arguments
    sys.argv = [
        "vietnamese_train.py",
        "--model_path", "bosonai/higgs-audio-v2-generation-3B-base",
        "--audio_tokenizer_path", "bosonai/higgs-audio-v2-tokenizer",
        "--train_data_dir", "/home/thuc/thuc/voice/train-higgs-audio-vi/vietnamese_training_data_fast",
        "--output_dir", "./output/vietnamese_higgs_model",
        "--logging_dir", "./logs/vietnamese_training",
        "--task_type", "single_speaker_smart_voice",
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "2",
        "--learning_rate", "1e-4",
        "--warmup_steps", "1000",
        "--logging_steps", "10",
        "--save_steps", "1000",
        "--use_lora",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--fp16",
        "--seed", "42",
        "--report_to", "tensorboard"
    ]
    
    # Run the training
    train_main()
