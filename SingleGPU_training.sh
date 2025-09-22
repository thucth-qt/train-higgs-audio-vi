#!/usr/bin/env bash
# Single GPU training script for Higgs Audio v2 POC
# Usage: ./SingleGPU_training.sh [fp16|bf16]
# Default: fp16

set -e

# Activate the local virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

# Disable wandb logging
export WANDB_DISABLED=true

# Precision selection
PRECISION=${1:-fp16}
if [[ "$PRECISION" == "bf16" ]]; then
  PRECISION_FLAG="--bf16"
  echo "[INFO] Using bfloat16 (bf16) precision. Make sure your GPU supports it!"
elif [[ "$PRECISION" == "fp16" ]]; then
  PRECISION_FLAG="--fp16"
  echo "[INFO] Using float16 (fp16) precision."
else
  echo "[ERROR] Unknown precision: $PRECISION. Use 'fp16' or 'bf16'."
  exit 1
fi

# Run the training script
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights \
  --audio_tokenizer_path /root/data/higgs/weights \
  --train_data_dir /root/data/higgs/train-higgs-audio-vi/higgs_training_data_mini \
  --output_dir /root/data/higgs/train-higgs-audio-vi/output_poc \
  --per_device_train_batch_size 4 \
  --num_train_epochs 1 \
  $PRECISION_FLAG \
  --logging_steps 10 \
  --save_steps 10 \
  --eval_steps 10 \
  --report_to none
