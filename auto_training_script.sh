#!/usr/bin/env bash
# Auto-generated training script for Higgs Audio v2
# Generated based on hardware: {'gpu_available': True, 'gpu_count': 1, 'cpu_cores': 30, 'total_ram_gb': 106.105335808, 'gpu_name': 'NVIDIA GeForce RTX 4090', 'gpu_memory_gb': 50.86543872, 'bf16_supported': True}
# Dataset samples: 52842

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
    print(f'GPU Memory: {mem_gb:.1f} GB')
else:
    print('WARNING: CUDA not available')
"

# Pre-training validation
echo "[INFO] Running pre-training validation..."
python3 validate_setup.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir ${1:-/root/data/higgs/balanced_tts_dataset_higgs_mini}

if [ $? -ne 0 ]; then
    echo "[ERROR] Pre-training validation failed"
    exit 1
fi

# Training command
echo "[INFO] Starting training with optimized configuration..."
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir ${1:-/root/data/higgs/balanced_tts_dataset_higgs_mini} \
  --output_dir ${2:-./runs/auto_generated_training} \
  --task_type single_speaker_smart_voice \
  --per_device_train_batch_size 3 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 2e-05 \
  --warmup_steps 500 \
  --logging_steps 50 \
  --save_steps 200 \
  --eval_steps 100 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine_with_restarts \
  --bf16 \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  --report_to tensorboard \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[SUCCESS] Training completed!"
else
    echo "[ERROR] Training failed"
    exit 1
fi
