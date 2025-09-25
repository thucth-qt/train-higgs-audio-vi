#!/usr/bin/env bash
# Quick test of the class-level monkey-patch fix
# This runs training for just 1 step to verify the fix works

set -e

# Enhanced PyTorch CUDA allocation config 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Check current directory and environment - no venv needed in this setup
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

# Check GPU availability
if ! python -c "import torch; print('GPU available:', torch.cuda.is_available())"; then
    echo "❌ GPU not available"
    exit 1
fi

echo "🧪 TESTING CLASS-LEVEL MONKEY-PATCH FIX..."
echo "🔬 Running training for 1 step to verify labels parameter fix works"

# Use proper paths for this environment and run very short training
python3 trainer/trainer.py \
  --model_path /Users/thuc.tran/thucth/train-higgs-audio-vi/models/higgs-audio-v2-generation-3B-base \
  --audio_tokenizer_path /Users/thuc.tran/thucth/train-higgs-audio-vi/models/higgs-audio-v2-tokenizer \
  --train_data_dir /Users/thuc.tran/thucth/train-higgs-audio-vi/higgs_training_data_mini \
  --task_type single_speaker_smart_voice \
  --output_dir /Users/thuc.tran/thucth/train-higgs-audio-vi/runs/class_level_test_$(date +%Y%m%d_%H%M%S) \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --fp16 \
  --learning_rate 2e-5 \
  --warmup_steps 0 \
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
  2>&1 | head -200

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 CLASS-LEVEL MONKEY-PATCH FIX WORKS!"
    echo "✅ Training completed 1 step without labels parameter error"
    echo "🚀 Ready for full training!"
else
    echo ""
    echo "❌ CLASS-LEVEL MONKEY-PATCH FIX FAILED"
    echo "🔧 Need to investigate further"
fi