#!/usr/bin/env bash
# Single GPU training script for Higgs Audio v2 on Vietnamese dataset (with LoRA)
# Usage: ./SingleGPU_training_vn_lora.sh [fp16|bf16]
# Default: fp16
# Compatible with new HiggsAudioTokenizer (audio tokenizer loads on CPU, only model on GPU)

set -e

# Set PyTorch CUDA allocation config to reduce fragmentation (recommended for large models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
# Optionally, reduce CPU contention
export OMP_NUM_THREADS=1

# Activate the local virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

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
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir /root/data/higgs/balanced_tts_dataset_higgs_mini \
  --output_dir /root/data/higgs/train-higgs-audio-vi/runs/2_output_vn_lora_mini \
  --per_device_train_batch_size 3 \
  --num_train_epochs 1 \
  $PRECISION_FLAG \
  --use_lora \
  --logging_steps 100 \
  --save_steps 100 \
  --eval_steps 100 \
  --report_to tensorboard
