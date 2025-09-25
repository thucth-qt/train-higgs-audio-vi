#!/usr/bin/env bash
# Zero-Shot Voice Cloning Training Script for Higgs Audio v2
# Usage: ./ZeroShotVoiceCloning_training.sh [fp16|bf16]

set -e

# Enhanced PyTorch CUDA allocation config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Check if voice cloning dataset exists
VOICE_CLONING_DATASET="/root/data/higgs/balanced_tts_voice_cloning_dataset_zero_shot"
if [ ! -d "$VOICE_CLONING_DATASET" ]; then
    echo "❌ Voice cloning dataset not found at: $VOICE_CLONING_DATASET"
    echo "🔧 Please generate it first with:"
    echo "   python tools/generate_voice_cloning_dataset.py --task_type zero_shot_voice_cloning --input_dir ./higgs_training_data_mini --output_dir $VOICE_CLONING_DATASET"
    exit 1
fi

# Activate the virtual environment
source /root/data/higgs/train-higgs-audio-vi/.venv/bin/activate

# Precision selection with validation
PRECISION=${1:-fp16}
if [[ "$PRECISION" == "bf16" ]]; then
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
  echo "[ERROR] Invalid precision: $PRECISION. Use 'fp16' or 'bf16'"
  exit 1
fi

echo "[INFO] Checking GPU memory availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {total_memory:.1f} GB')
    if total_memory < 12:
        print('WARNING: Low GPU memory detected. Consider reducing batch size.')
else:
    print('ERROR: CUDA not available')
    exit(1)
"

echo "[INFO] Running pre-training validation..."
if ! python3 validate_setup.py; then
    echo "❌ Pre-training validation failed"
    exit 1
fi
echo "✓ Pre-training validation passed!"

echo "[INFO] Starting Zero-Shot Voice Cloning training with enhanced error handling..."

# Create output directory with timestamp
OUTPUT_DIR="/root/data/higgs/train-higgs-audio-vi/runs/zero_shot_voice_cloning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "[INFO] Created output directory: $OUTPUT_DIR"

# Run training with comprehensive error handling
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir "$VOICE_CLONING_DATASET" \
  --task_type zero_shot_voice_cloning \
  --ref_audio_in_system_message \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --warmup_steps 200 \
  --logging_steps 25 \
  --save_steps 500 \
  --eval_steps 250 \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42 \
  $PRECISION_FLAG \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine_with_restarts \
  --report_to tensorboard \
  --logging_dir "./logs/zero_shot_voice_cloning" \
  2>&1 | tee "$OUTPUT_DIR/training.log"

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 Zero-Shot Voice Cloning Training Completed Successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Training logs: $OUTPUT_DIR/training.log"
    echo "🎤 Ready for voice cloning inference!"
else
    echo ""
    echo "❌ Training failed. Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi