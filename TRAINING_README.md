# Higgs Audio v2 Training Implementation - Enhanced Version

This repository contains a comprehensive training implementation for Higgs Audio v2 TTS (Text-to-Speech) model with advanced error handling, memory optimization, and production-ready features.

## 🚀 Quick Start

### 1. Pre-training Validation
```bash
# Validate your setup before training
./validate_setup.py \
  --model_path /path/to/higgs-audio-v2-base \
  --audio_tokenizer_path /path/to/higgs-audio-tokenizer \
  --train_data_dir /path/to/training/data
```

### 2. Generate Optimal Configuration
```bash
# Auto-generate training configuration based on your hardware
./generate_training_config.py \
  --data_dir /path/to/training/data \
  --task_type single_speaker_smart_voice \
  --output_script my_training_script.sh

# Run the generated script
./my_training_script.sh
```

### 3. Manual Training
```bash
# LoRA training (recommended for most users)
./SingleGPU_training_vn_lora.sh fp16

# Full fine-tuning (requires high-end GPU)
./SingleGPU_training_vn_full.sh fp16
```

## 🛠 Enhanced Features

### ✅ **Robust Error Handling**
- **Dataset Validation**: Pre-training integrity checks for audio files and transcripts
- **Audio Loading Fallbacks**: Multiple audio loading strategies (torchaudio → librosa → soundfile)
- **Memory Management**: Automatic CUDA OOM recovery and cleanup
- **Graceful Degradation**: Skip corrupted samples without breaking training

### ✅ **Memory Optimization**
- **Gradient Checkpointing**: Enabled by default for memory efficiency
- **Smart Batch Sizing**: Auto-adjustment based on GPU memory
- **Memory Monitoring**: Real-time GPU/CPU memory tracking
- **Dynamic Memory Cleanup**: Periodic cache clearing during training

### ✅ **Training Stability**
- **Mixed Precision**: Support for both FP16 and BF16 with hardware detection
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing with restarts
- **Early Stopping**: Automatic best model selection

### ✅ **Production Features**
- **Resume Training**: Automatic checkpoint detection and resumption
- **Configuration Generation**: Hardware-optimized training configs
- **Comprehensive Logging**: Detailed progress and error reporting
- **Validation Scripts**: Pre-training setup verification

## 📋 System Requirements

### Minimum Requirements
- **GPU**: 16GB VRAM (RTX 4080, V100)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space
- **CUDA**: 11.8+ with compatible PyTorch

### Recommended Requirements
- **GPU**: 24GB+ VRAM (RTX 4090, A100)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB SSD
- **CUDA**: 12.1+ with latest PyTorch

## 🎯 Task Types

### 1. Single Speaker Smart Voice
```bash
python3 trainer/trainer.py \
  --task_type single_speaker_smart_voice \
  --train_data_dir /path/to/single_speaker_data
```

### 2. Zero-shot Voice Cloning
```bash
python3 trainer/trainer.py \
  --task_type zero_shot_voice_cloning \
  --ref_audio_in_system_message \
  --train_data_dir /path/to/voice_cloning_data
```

### 3. Multi-speaker Training
```bash
python3 trainer/trainer.py \
  --task_type multi_speaker_smart_voice \
  --train_data_dir /path/to/multispeaker_data
```

## ⚙️ Configuration Options

### LoRA Parameters
```bash
--use_lora \
--lora_rank 16 \          # Higher rank = more parameters (8, 16, 32, 64)
--lora_alpha 32 \         # Usually 2x the rank
--lora_dropout 0.1        # Dropout for regularization
```

### Memory Optimization
```bash
--per_device_train_batch_size 2 \     # Adjust based on GPU memory
--gradient_accumulation_steps 8 \     # Maintain effective batch size
--gradient_checkpointing \            # Enable memory saving
--dataloader_num_workers 0            # Avoid multiprocessing issues
```

### Training Hyperparameters
```bash
--learning_rate 2e-5 \                # Lower for stability
--warmup_steps 200 \                  # Gradual learning rate increase
--weight_decay 0.01 \                 # L2 regularization
--max_grad_norm 1.0                   # Gradient clipping
```

## 📊 Monitoring and Logging

### TensorBoard Integration
```bash
# View training progress
tensorboard --logdir ./logs/

# Key metrics:
# - train/loss: Training loss progression
# - train/learning_rate: Learning rate schedule
# - system/gpu_memory_gb: GPU memory usage
# - system/peak_memory_gb: Peak memory consumption
```

### Memory Monitoring
The trainer automatically logs:
- GPU memory allocation and reserved
- CPU memory usage
- Peak memory consumption
- Memory cleanup events

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory (OOM)
```bash
# Solutions (in order of preference):
1. Reduce batch size: --per_device_train_batch_size 1
2. Increase gradient accumulation: --gradient_accumulation_steps 16
3. Enable gradient checkpointing: --gradient_checkpointing
4. Use smaller LoRA rank: --lora_rank 8
```

#### 2. Audio Loading Errors
```bash
# The trainer automatically handles:
- Corrupted audio files (skipped)
- Format conversion issues (multiple fallbacks)
- Sample rate mismatches (automatic resampling)
- Empty or very short audio files
```

#### 3. Dataset Issues
```bash
# Run validation before training:
./validate_setup.py --train_data_dir /your/data/path

# Common fixes:
- Ensure metadata.json exists and is valid
- Check audio file paths are correct
- Verify transcript files exist
- Remove corrupted samples
```

#### 4. Model Loading Issues
```bash
# Ensure model paths are correct:
--model_path /path/to/higgs-audio-v2-generation-3B-base
--audio_tokenizer_path /path/to/higgs-audio-v2-tokenizer

# Check for required files:
- config.json
- pytorch_model.bin or model.safetensors
- tokenizer files
```

## 🔄 Resume Training

### Automatic Resume
```bash
# Find and resume from latest checkpoint
./resume_training.py \
  --output_dir /path/to/training/output \
  --execute
```

### Manual Resume
```bash
python3 trainer/trainer.py \
  --resume_from_checkpoint /path/to/checkpoint-1000 \
  [other training arguments]
```

## 📈 Performance Optimization

### Hardware-Specific Configurations

#### RTX 4090 (24GB)
```bash
--per_device_train_batch_size 2
--gradient_accumulation_steps 8
--use_lora --lora_rank 16
--fp16
```

#### A100 40GB
```bash
--per_device_train_batch_size 3
--gradient_accumulation_steps 4
--use_lora --lora_rank 32
--bf16
```

#### V100 16GB
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--use_lora --lora_rank 8
--fp16
```

## 📝 Dataset Format

### Required Structure
```
dataset/
├── metadata.json          # Dataset metadata
├── audio_file_001.wav     # Audio files
├── audio_file_001.txt     # Transcript files
├── audio_file_002.wav
├── audio_file_002.txt
└── ...
```

### Metadata Format
```json
{
  "dataset_info": {
    "total_samples": 1000,
    "total_duration": 3600,
    "sample_rate": 24000
  },
  "samples": [
    {
      "audio_file": "audio_file_001.wav",
      "transcript_file": "audio_file_001.txt",
      "audio_id": "speaker_001_001"
    }
  ]
}
```

## 🧪 Testing

### Quick Component Test
```bash
# Test trainer components
./test_trainer.py
```

### Full System Validation
```bash
# Comprehensive pre-training validation
./validate_setup.py \
  --model_path /path/to/model \
  --audio_tokenizer_path /path/to/tokenizer \
  --train_data_dir /path/to/data
```

## 🎨 Advanced Usage

### Custom Task Implementation
```python
# Extend HiggsAudioDataset for custom tasks
class CustomTaskDataset(HiggsAudioDataset):
    def _create_messages_for_task(self, sample, transcript):
        # Custom message creation logic
        messages = []
        # ... your implementation
        return messages
```

### Custom Loss Functions
```python
# Extend HiggsAudioTrainer for custom loss
class CustomTrainer(HiggsAudioTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss computation
        # ... your implementation
        return loss
```

## 📜 License and Attribution

This implementation extends the original Higgs Audio v2 model with enhanced training capabilities, error handling, and production features. Please refer to the original Higgs Audio licensing terms for usage restrictions.

## 🆘 Support

If you encounter issues:

1. **Check the logs**: Training logs contain detailed error information
2. **Run validation**: Use `validate_setup.py` to check your setup
3. **Monitor memory**: Use the built-in memory monitoring features
4. **Start small**: Test with a small dataset first
5. **Check hardware**: Ensure your GPU has sufficient memory

For optimal results, start with the auto-generated configuration and adjust based on your specific needs and hardware constraints.