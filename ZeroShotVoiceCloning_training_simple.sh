#!/usr/bin/env bash
# Simple Zero-Shot Voice Cloning Training Script - Essential Fixes Only
# Usage: ./ZeroShotVoiceCloning_training_simple.sh

set -e

# Basic PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# Check if voice cloning dataset exists
VOICE_CLONING_DATASET="/root/data/higgs/balanced_tts_voice_cloning_dataset_zero_shot"
if [ ! -d "$VOICE_CLONING_DATASET" ]; then
    echo "❌ Voice cloning dataset not found at: $VOICE_CLONING_DATASET"
    exit 1
fi

# Activate the virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

echo "[INFO] Starting Simple Zero-Shot Voice Cloning training..."
echo "[INFO] Using FP16 precision"

# Create output directory with timestamp
OUTPUT_DIR="/root/data/higgs/train-higgs-audio-vi/runs/zero_shot_voice_cloning_simple_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "[INFO] Created output directory: $OUTPUT_DIR"

# Run training with basic settings
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir "$VOICE_CLONING_DATASET" \
  --task_type zero_shot_voice_cloning \
  --ref_audio_in_system_message \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 250 \
  --eval_steps 125 \
  --use_lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --seed 42 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 0 \
  --max_grad_norm 0.5 \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --report_to tensorboard \
  --logging_dir "./logs/zero_shot_voice_cloning_simple" \
  --dataloader_pin_memory false \
  --remove_unused_columns false \
  2>&1 | tee "$OUTPUT_DIR/training.log"

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 Simple Zero-Shot Voice Cloning Training Completed Successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Training logs: $OUTPUT_DIR/training.log"
else
    echo ""
    echo "❌ Training failed. Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi