#!/usr/bin/env bash
# Multi-Speaker Voice Cloning Training Script for Higgs Audio v2  
# Usage: ./MultiSpeakerVoiceCloning_training.sh [fp16|bf16]

set -e

# Enhanced PyTorch CUDA allocation config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Check if multi-speaker voice cloning dataset exists
VOICE_CLONING_DATASET="/root/data/higgs/voice_cloning_dataset_multi_speaker"
if [ ! -d "$VOICE_CLONING_DATASET" ]; then
    echo "❌ Multi-speaker voice cloning dataset not found at: $VOICE_CLONING_DATASET"
    echo "🔧 Please generate it first with:"
    echo "   python tools/generate_voice_cloning_dataset.py --task_type multi_speaker_voice_cloning --input_dir ./higgs_training_data_mini --output_dir $VOICE_CLONING_DATASET"
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
    if total_memory < 16:
        print('WARNING: Multi-speaker training requires more memory. Consider reducing batch size.')
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

echo "[INFO] Starting Multi-Speaker Voice Cloning training with enhanced error handling..."

# Create output directory with timestamp
OUTPUT_DIR="/root/data/higgs/train-higgs-audio-vi/runs/multi_speaker_voice_cloning_$(date +%Y%m%d_%H%M%S)"

# Run training with comprehensive error handling
python3 trainer/trainer.py \
  --model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /root/data/higgs/weights/higgs-audio-v2-tokenizer \
  --train_data_dir "$VOICE_CLONING_DATASET" \
  --task_type multi_speaker_voice_cloning \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 8e-6 \
  --warmup_steps 300 \
  --logging_steps 20 \
  --save_steps 400 \
  --eval_steps 200 \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --seed 42 \
  $PRECISION_FLAG \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --max_grad_norm 0.5 \
  --weight_decay 0.005 \
  --lr_scheduler_type cosine_with_restarts \
  --report_to tensorboard \
  --logging_dir "./logs/multi_speaker_voice_cloning" \
  2>&1 | tee "$OUTPUT_DIR/training.log"

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 Multi-Speaker Voice Cloning Training Completed Successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Training logs: $OUTPUT_DIR/training.log"
    echo "🎭 Ready for multi-speaker voice cloning inference!"
else
    echo ""
    echo "❌ Training failed. Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi