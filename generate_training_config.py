#!/usr/bin/env python3
"""
Training configuration generator for Higgs Audio v2
Generates optimal training configurations based on hardware and dataset
"""

import argparse
import json
import logging
import torch
import psutil
import os
import stat
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_hardware():
    """Detect hardware specifications"""
    hardware_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_cores": psutil.cpu_count(),
        "total_ram_gb": psutil.virtual_memory().total / 1e9,
    }
    
    if hardware_info["gpu_available"]:
        gpu_props = torch.cuda.get_device_properties(0)
        hardware_info.update({
            "gpu_name": gpu_props.name,
            "gpu_memory_gb": gpu_props.total_memory / 1e9,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    
    return hardware_info

def analyze_dataset(data_dir):
    """Analyze dataset characteristics"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Dataset directory not found: {data_dir}")
        return {"samples": 0, "estimated_size_gb": 0}
    
    # Try to load metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            samples = metadata.get("samples", [])
            sample_count = len(samples)
            
            # Estimate dataset size
            total_duration = metadata.get("dataset_info", {}).get("total_duration", 0)
            estimated_size_gb = total_duration * 24000 * 2 / 1e9  # Rough estimate for 24kHz 16-bit audio
            
            return {
                "samples": sample_count,
                "total_duration_hours": total_duration / 3600,
                "estimated_size_gb": estimated_size_gb,
                "avg_duration": metadata.get("dataset_info", {}).get("avg_duration", 0),
            }
            
        except Exception as e:
            logger.warning(f"Could not parse metadata: {e}")
    
    # Fallback: count audio files
    audio_files = list(data_path.glob("*.wav")) + list(data_path.glob("*.mp3"))
    return {
        "samples": len(audio_files),
        "estimated_size_gb": len(audio_files) * 0.1,  # Very rough estimate
    }

def generate_training_config(hardware_info, dataset_info, task_type="single_speaker_smart_voice"):
    """Generate optimal training configuration"""
    
    config = {
        "model_path": "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base",
        "audio_tokenizer_path": "/root/data/higgs/weights/higgs-audio-v2-tokenizer",
        "task_type": task_type,
        "seed": 42,
        "logging_steps": 50,
        "save_steps": 200,
        "eval_steps": 100,
        "save_total_limit": 3,
        "report_to": "tensorboard",
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine_with_restarts",
    }
    
    # GPU memory based configurations
    gpu_memory = hardware_info.get("gpu_memory_gb", 0)
    
    if gpu_memory >= 80:  # A100 80GB or H100
        config.update({
            "per_device_train_batch_size": 4,
            "learning_rate": 3e-5,
            "use_lora": False,  # Can do full fine-tuning
            "gradient_accumulation_steps": 2,
        })
        logger.info("Configuration: High-end GPU - Full fine-tuning")
        
    elif gpu_memory >= 40:  # A100 40GB
        config.update({
            "per_device_train_batch_size": 3,
            "learning_rate": 2e-5,
            "use_lora": True,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 4,
        })
        logger.info("Configuration: High-memory GPU - LoRA with larger rank")
        
    elif gpu_memory >= 24:  # RTX 4090, RTX 6000 Ada
        config.update({
            "per_device_train_batch_size": 2,
            "learning_rate": 2e-5,
            "use_lora": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 8,
        })
        logger.info("Configuration: Mid-range GPU - Standard LoRA")
        
    elif gpu_memory >= 16:  # RTX 4080, V100
        config.update({
            "per_device_train_batch_size": 1,
            "learning_rate": 1e-5,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "gradient_accumulation_steps": 16,
        })
        logger.info("Configuration: Lower-memory GPU - Minimal LoRA")
        
    else:
        logger.warning("GPU memory too low for training this model")
        config.update({
            "per_device_train_batch_size": 1,
            "learning_rate": 5e-6,
            "use_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 32,
        })
    
    # Precision selection
    if hardware_info.get("bf16_supported", False):
        config["bf16"] = True
        logger.info("Using BFloat16 precision")
    else:
        config["fp16"] = True
        logger.info("Using Float16 precision")
    
    # Dataset size based configurations
    sample_count = dataset_info.get("samples", 0)
    
    if sample_count > 10000:
        config["num_train_epochs"] = 1  # Large dataset, fewer epochs
        config["warmup_steps"] = 500
    elif sample_count > 1000:
        config["num_train_epochs"] = 2  # Medium dataset
        config["warmup_steps"] = 200
    else:
        config["num_train_epochs"] = 3  # Small dataset, more epochs
        config["warmup_steps"] = 100
    
    return config

def generate_training_script(config, output_path):
    """Generate a training script with the configuration"""
    
    script_content = f'''#!/usr/bin/env bash
# Auto-generated training script for Higgs Audio v2
# Generated based on hardware: {config.get("_hardware_info", "Unknown")}
# Dataset samples: {config.get("_dataset_samples", "Unknown")}

set -e

# Enhanced PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Activate virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

# GPU memory check
echo "[INFO] Checking GPU memory..."
python3 -c "
import torch
if torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU Memory: {{mem_gb:.1f}} GB')
else:
    print('WARNING: CUDA not available')
"

# Pre-training validation
echo "[INFO] Running pre-training validation..."
python3 validate_setup.py \\
  --model_path {config['model_path']} \\
  --audio_tokenizer_path {config['audio_tokenizer_path']} \\
  --train_data_dir ${{1:-/root/data/higgs/balanced_tts_dataset_higgs_mini}}

if [ $? -ne 0 ]; then
    echo "[ERROR] Pre-training validation failed"
    exit 1
fi

# Training command
echo "[INFO] Starting training with optimized configuration..."
python3 trainer/trainer.py \\
  --model_path {config['model_path']} \\
  --audio_tokenizer_path {config['audio_tokenizer_path']} \\
  --train_data_dir ${{1:-/root/data/higgs/balanced_tts_dataset_higgs_mini}} \\
  --output_dir ${{2:-./runs/auto_generated_training}} \\
  --task_type {config['task_type']} \\
  --per_device_train_batch_size {config['per_device_train_batch_size']} \\
  --gradient_accumulation_steps {config.get('gradient_accumulation_steps', 1)} \\
  --num_train_epochs {config['num_train_epochs']} \\
  --learning_rate {config['learning_rate']} \\
  --warmup_steps {config['warmup_steps']} \\
  --logging_steps {config['logging_steps']} \\
  --save_steps {config['save_steps']} \\
  --eval_steps {config['eval_steps']} \\
  --max_grad_norm {config['max_grad_norm']} \\
  --weight_decay {config['weight_decay']} \\
  --lr_scheduler_type {config['lr_scheduler_type']} \\
'''
    
    # Add precision flag
    if config.get("bf16", False):
        script_content += "  --bf16 \\\n"
    elif config.get("fp16", False):
        script_content += "  --fp16 \\\n"
    
    # Add LoRA configuration if enabled
    if config.get("use_lora", False):
        script_content += f'''  --use_lora \\
  --lora_rank {config.get('lora_rank', 16)} \\
  --lora_alpha {config.get('lora_alpha', 32)} \\
  --lora_dropout {config.get('lora_dropout', 0.1)} \\
'''
    
    script_content += f'''  --gradient_checkpointing \\
  --dataloader_num_workers {config['dataloader_num_workers']} \\
  --remove_unused_columns {str(config['remove_unused_columns']).lower()} \\
  --report_to {config['report_to']} \\
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# Check training result
if [ ${{PIPESTATUS[0]}} -eq 0 ]; then
    echo "[SUCCESS] Training completed!"
else
    echo "[ERROR] Training failed"
    exit 1
fi
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    import stat
    os.chmod(output_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

def main():
    parser = argparse.ArgumentParser(description="Generate optimal training configuration")
    parser.add_argument("--data_dir", type=str, 
                       default="/root/data/higgs/balanced_tts_dataset_higgs_mini",
                       help="Training data directory")
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice",
                       choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", 
                               "multi_speaker_smart_voice", "multi_speaker_voice_cloning"])
    parser.add_argument("--output_config", type=str, default="auto_training_config.json",
                       help="Output configuration file")
    parser.add_argument("--output_script", type=str, default="auto_training_script.sh",
                       help="Output training script")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HIGGS AUDIO V2 TRAINING CONFIG GENERATOR")
    logger.info("=" * 60)
    
    # Detect hardware
    logger.info("Detecting hardware...")
    hardware_info = detect_hardware()
    
    logger.info(f"Hardware detected:")
    logger.info(f"  GPU: {hardware_info.get('gpu_name', 'N/A')}")
    logger.info(f"  GPU Memory: {hardware_info.get('gpu_memory_gb', 0):.1f} GB")
    logger.info(f"  CPU Cores: {hardware_info['cpu_cores']}")
    logger.info(f"  RAM: {hardware_info['total_ram_gb']:.1f} GB")
    logger.info(f"  BF16 Support: {hardware_info.get('bf16_supported', False)}")
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    dataset_info = analyze_dataset(args.data_dir)
    
    logger.info(f"Dataset info:")
    logger.info(f"  Samples: {dataset_info['samples']:,}")
    logger.info(f"  Estimated size: {dataset_info.get('estimated_size_gb', 0):.1f} GB")
    if 'total_duration_hours' in dataset_info:
        logger.info(f"  Total duration: {dataset_info['total_duration_hours']:.1f} hours")
    
    # Generate configuration
    logger.info("Generating optimal configuration...")
    config = generate_training_config(hardware_info, dataset_info, args.task_type)
    
    # Add metadata
    config["_hardware_info"] = hardware_info
    config["_dataset_info"] = dataset_info
    config["_dataset_samples"] = dataset_info['samples']
    
    # Save configuration
    with open(args.output_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Configuration saved to: {args.output_config}")
    
    # Generate training script
    generate_training_script(config, args.output_script)
    logger.info(f"✓ Training script generated: {args.output_script}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Task type: {config['task_type']}")
    logger.info(f"Batch size: {config['per_device_train_batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Epochs: {config['num_train_epochs']}")
    logger.info(f"LoRA enabled: {config.get('use_lora', False)}")
    if config.get('use_lora', False):
        logger.info(f"  LoRA rank: {config.get('lora_rank', 'N/A')}")
        logger.info(f"  LoRA alpha: {config.get('lora_alpha', 'N/A')}")
    logger.info(f"Precision: {'BF16' if config.get('bf16') else 'FP16'}")
    logger.info(f"Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    
    logger.info("=" * 60)
    logger.info(f"To start training, run: ./{args.output_script} [data_dir] [output_dir]")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()