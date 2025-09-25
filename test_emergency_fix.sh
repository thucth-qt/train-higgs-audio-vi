#!/usr/bin/env bash
# Emergency Test Script for Higgs Audio v2 Training with Monkey-Patch Fix
# Usage: ./test_emergency_fix.sh [fp16|bf16]
# Default: fp16
# This script tests the emergency monkey-patch solution for the labels parameter issue

set -e

# Enhanced PyTorch CUDA allocation config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
# Reduce CPU contention and optimize for single GPU training
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Activate the local virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

# Precision selection with GPU support validation
PRECISION=${1:-fp16}
if [[ "$PRECISION" == "bf16" ]]; then
  # Check if GPU supports bfloat16
  if python3 -c "import torch; exit(0 if torch.cuda.is_bf16_supported() else 1)" 2>/dev/null; then
    PRECISION_FLAG="--bf16"
    echo "[INFO] Using bfloat16 (bf16) precision - GPU support confirmed"
  else
    echo "[WARNING] GPU does not support bf16, falling back to fp16"
    PRECISION_FLAG="--fp16"
  fi
elif [[ "$PRECISION" == "fp16" ]]; then
  PRECISION_FLAG="--fp16"
  echo "[INFO] Using float16 (fp16) precision"
else
  echo "[ERROR] Unknown precision: $PRECISION. Use 'fp16' or 'bf16'"
  exit 1
fi

# Memory optimization check
echo "[INFO] Checking GPU memory availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU Memory: {mem_gb:.1f} GB')
    if mem_gb < 16:
        print('WARNING: GPU has less than 16GB memory. Consider reducing batch size.')
else:
    print('WARNING: CUDA not available')
"

# Pre-training validation
echo "[INFO] Running pre-training validation..."
python3 validate_setup.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir /root/data/higgs/balanced_tts_dataset_higgs_mini

if [ $? -ne 0 ]; then
    echo "[ERROR] Pre-training validation failed. Please fix issues before training."
    exit 1
fi

echo "[SUCCESS] Pre-training validation passed!"

# Run the training script with emergency monkey-patch fix
echo "[INFO] 🚑 Testing EMERGENCY MONKEY-PATCH fix for labels parameter issue..."
echo "[INFO] This script runs training for 3 steps to test the fix quickly."
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir /root/data/higgs/balanced_tts_dataset_higgs_mini \
  --task_type single_speaker_smart_voice \
  --output_dir /root/data/higgs/train-higgs-audio-vi/runs/emergency_test_$(date +%Y%m%d_%H%M%S) \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --max_steps 3 \
  $PRECISION_FLAG \
  --learning_rate 2e-5 \
  --warmup_steps 1 \
  --logging_steps 1 \
  --save_steps 999999 \
  --eval_steps 999999 \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine_with_restarts \
  --seed 42 \
  --report_to none \
  2>&1 | tee emergency_test_$(date +%Y%m%d_%H%M%S).log

# Check training completion
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "🎉 [SUCCESS] Emergency monkey-patch fix WORKS! Training completed successfully!"
    echo "[INFO] The labels parameter issue has been resolved."
    echo "[INFO] You can now run full training with confidence."
    echo "[INFO] Test logs saved to emergency_test_$(date +%Y%m%d_%H%M%S).log"
else
    echo "❌ [ERROR] Emergency monkey-patch fix failed with exit code ${PIPESTATUS[0]}"
    echo "[INFO] Please check the debug logs for more details."
    echo "[INFO] Test logs saved to emergency_test_$(date +%Y%m%d_%H%M%S).log"
    exit 1
fi