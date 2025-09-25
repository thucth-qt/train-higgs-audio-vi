#!/usr/bin/env bash
# Single GPU training script for Higgs Audio v2 on Vietnamese dataset (with LoRA)
# Usage: ./SingleGPU_training_vn_lora.sh [fp16|bf16]
# Default: fp16
# Compatible with new HiggsAudioTokenizer (audio tokenizer loads on CPU, only model on GPU)

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

# Precision selection with validation
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

# Run the training script with enhanced monitoring
echo "[INFO] Starting Higgs Audio v2 LoRA training with enhanced error handling..."
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir /root/data/higgs/balanced_tts_dataset_higgs_mini \
  --output_dir /root/data/higgs/train-higgs-audio-vi/runs/2_output_vn_lora_mini \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1 \
  $PRECISION_FLAG \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --learning_rate 2e-5 \
  --warmup_steps 100 \
  --logging_steps 50 \
  --save_steps 200 \
  --eval_steps 100 \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine_with_restarts \
  --report_to tensorboard \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# Check training completion
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully!"
    echo "[INFO] Logs saved to training_$(date +%Y%m%d_%H%M%S).log"
else
    echo "[ERROR] Training failed with exit code ${PIPESTATUS[0]}"
    exit 1
fi
