#!/bin/bash
# Setup Training Environment for Vietnamese TTS

echo "🔧 Setting up training environment for Vietnamese TTS..."

# Install required packages for training
echo "📦 Installing training dependencies..."

# Install PEFT for LoRA
uv pip install peft==0.16.0

# Install other missing dependencies that might be needed
uv pip install accelerate
uv pip install tensorboard

echo "✅ Training environment setup complete!"
echo ""
echo "🚀 You can now start training with:"
echo "  bash train_vietnamese.sh"
echo ""
echo "Or manually with:"
echo "  python trainer/trainer.py --train_data_dir /home/thuc/thuc/voice/train-higgs-audio-vi/vietnamese_training_data_fast --use_lora --fp16"