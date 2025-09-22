#!/bin/bash
# Vietnamese TTS Training Script for Higgs Audio v2
# Run this script to start training with optimized settings

echo "🚀 Starting Vietnamese TTS Training with Higgs Audio v2"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "trainer/trainer.py" ]; then
    echo "❌ Error: trainer/trainer.py not found. Please run from the project root."
    exit 1
fi

# Check if data directory exists
DATA_DIR="/root/data/higgs/train-higgs-audio-vi/vietnamese_training_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Training data directory not found: $DATA_DIR"
    echo "Please run the data preprocessing first."
    exit 1
fi

# Create output directories
mkdir -p "./output/vietnamese_higgs_model"
mkdir -p "./logs/vietnamese_training"

echo "📊 Dataset Information:"
echo "  Training data: $DATA_DIR"
echo "  Output directory: ./output/vietnamese_higgs_model"
echo "  Logs directory: ./logs/vietnamese_training"

# Training configuration
MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH="bosonai/higgs-audio-v2-tokenizer"
OUTPUT_DIR="./output/vietnamese_higgs_model"
LOGGING_DIR="./logs/vietnamese_training"

echo ""
echo "🔧 Training Configuration (Memory Optimized):"
echo "  Model: $MODEL_PATH"
echo "  Audio Tokenizer: $AUDIO_TOKENIZER_PATH"
echo "  Task: Vietnamese TTS (single speaker smart voice)"
echo "  LoRA: Enabled (rank=4, alpha=8) - Further reduced for memory"
echo "  Epochs: 5"
echo "  Batch size: 1 (optimized for 20GB GPU)"
echo "  Gradient accumulation: 8 steps (effective batch=8)"
echo "  Learning rate: 1e-4"
echo "  Mixed Precision: FP16"
echo "  Sequence length: 1536 (reduced)"
echo "  Memory optimizations: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  DataLoader workers: 0 (CUDA multiprocessing disabled)"
echo ""

# Ask for confirmation
read -p "Do you want to start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo "🏃‍♂️ Starting training..."
echo ""

# CRITICAL: Set CUDA memory allocation strategy
echo "🔧 Setting CUDA memory optimization..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
echo "   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Clear GPU cache before training
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Run the training with memory-optimized parameters for 20GB GPU
python trainer/trainer.py \
    --model_path "$MODEL_PATH" \
    --audio_tokenizer_path "$AUDIO_TOKENIZER_PATH" \
    --train_data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR" \
    --task_type "single_speaker_smart_voice" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --use_lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --fp16 \
    --dataloader_num_workers 0 \
    --max_length 1536 \
    --seed 42 \
    --report_to tensorboard

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Logs saved to: $LOGGING_DIR"
    echo ""
    echo "To view training logs:"
    echo "  tensorboard --logdir $LOGGING_DIR"
    echo ""
    echo "To test the model, use the generation script:"
    echo "  bash generate.sh"
else
    echo ""
    echo "❌ Training failed. Check the logs for errors."
    echo "📊 Logs location: $LOGGING_DIR"
fi