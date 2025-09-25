# 🎤 Voice Cloning Training Guide for Higgs Audio v2

## 📋 Overview

This guide helps you set up and train voice cloning models using Higgs Audio v2. There are two main voice cloning tasks supported:

1. **Zero-Shot Voice Cloning**: Clone a voice using just one reference audio sample
2. **Multi-Speaker Voice Cloning**: Clone multiple voices and switch between them in generation

## 🚀 Quick Start

### Step 1: Generate Voice Cloning Dataset

First, you need to convert your existing dataset to the voice cloning format:

#### For Zero-Shot Voice Cloning:
```bash
python tools/generate_voice_cloning_dataset.py \
  --task_type zero_shot_voice_cloning \
  --input_dir ./higgs_training_data_mini \
  --output_dir ./voice_cloning_dataset_zero_shot
```

#### For Multi-Speaker Voice Cloning:
```bash
python tools/generate_voice_cloning_dataset.py \
  --task_type multi_speaker_voice_cloning \
  --input_dir ./higgs_training_data_mini \
  --output_dir ./voice_cloning_dataset_multi_speaker
```

### Step 2: Start Training

#### Zero-Shot Voice Cloning Training:
```bash
chmod +x ZeroShotVoiceCloning_training.sh
./ZeroShotVoiceCloning_training.sh fp16
```

#### Multi-Speaker Voice Cloning Training:
```bash
chmod +x MultiSpeakerVoiceCloning_training.sh
./MultiSpeakerVoiceCloning_training.sh fp16
```

## 📊 Dataset Format Requirements

### Zero-Shot Voice Cloning Dataset Format

Each sample in the metadata.json needs:
```json
{
  "id": "sample_001",
  "audio_file": "sample_001.wav",
  "transcript_file": "sample_001.txt",
  "ref_audio_file": "ref_sample_001.wav",
  "ref_transcript": "This is the reference audio transcript",
  "speaker_id": "speaker_1",
  "task_type": "zero_shot_voice_cloning"
}
```

### Multi-Speaker Voice Cloning Dataset Format

Each sample needs multiple speaker references:
```json
{
  "id": "sample_001", 
  "audio_file": "sample_001.wav",
  "transcript_file": "sample_001.txt",
  "ref_speakers": [
    {
      "speaker_tag": "[SPEAKER0]",
      "speaker_id": "speaker_1",
      "ref_audio_file": "ref_speaker1.wav",
      "ref_transcript": "Reference audio for speaker 1"
    },
    {
      "speaker_tag": "[SPEAKER1]", 
      "speaker_id": "speaker_2",
      "ref_audio_file": "ref_speaker2.wav",
      "ref_transcript": "Reference audio for speaker 2"
    }
  ],
  "task_type": "multi_speaker_voice_cloning"
}
```

## 🔧 Training Configuration

### Zero-Shot Voice Cloning Settings:
- **Batch Size**: 1 (voice cloning requires more memory per sample)
- **Learning Rate**: 1e-5 (lower for fine-tuning voice characteristics)
- **LoRA Rank**: 16 (focused adaptation)
- **LoRA Alpha**: 32
- **Epochs**: 3
- **Reference Audio**: Uses `ref_audio_in_system_message` flag

### Multi-Speaker Voice Cloning Settings:
- **Batch Size**: 1 (complex multi-speaker samples)
- **Learning Rate**: 8e-6 (even lower for multi-speaker stability)
- **LoRA Rank**: 32 (higher for multi-speaker complexity) 
- **LoRA Alpha**: 64
- **Epochs**: 4
- **Multiple References**: Handles 2-4 reference speakers per sample

## 💾 Memory Requirements

### Minimum Requirements:
- **Zero-Shot Voice Cloning**: 16GB GPU memory
- **Multi-Speaker Voice Cloning**: 20GB+ GPU memory (more complex)

### Memory Optimization:
- Use gradient checkpointing (enabled by default)
- Batch size of 1 for voice cloning tasks
- LoRA fine-tuning reduces memory vs full fine-tuning

## 📁 Output Structure

Training outputs will be saved to timestamped directories:
```
runs/
├── zero_shot_voice_cloning_20250925_123045/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── training.log
│   └── trainer_state.json
└── multi_speaker_voice_cloning_20250925_124530/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── training.log
    └── trainer_state.json
```

## 🎯 Best Practices

### Data Quality:
1. **High-quality reference audio** (clean, clear, representative)
2. **Consistent speaker representation** across reference samples
3. **Diverse content** in reference transcripts
4. **Balanced speaker distribution** for multi-speaker tasks

### Training Tips:
1. **Start with fewer epochs** and monitor overfitting
2. **Use validation set** if available (10-20% of data)
3. **Monitor loss curves** in TensorBoard
4. **Save checkpoints frequently** for recovery

### Reference Audio Selection:
- **Length**: 3-10 seconds per reference sample
- **Quality**: Clean, noise-free audio
- **Content**: Natural speech, not artificial or distorted
- **Emotion**: Match the target emotion if specific

## 🔍 Troubleshooting

### Common Issues:

1. **Out of Memory Error**:
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use fp16 instead of bf16

2. **Poor Voice Quality**:
   - Check reference audio quality
   - Increase training epochs
   - Verify dataset format

3. **Training Instability**:
   - Lower learning rate
   - Increase warmup steps
   - Check gradient norm values

4. **Dataset Generation Fails**:
   - Ensure input dataset has multiple speakers
   - Check metadata.json format
   - Verify audio files exist

## 📊 Monitoring Training

### Key Metrics to Watch:
- **Loss convergence**: Should decrease steadily
- **Gradient norm**: Should stay stable (not exploding)
- **Memory usage**: Monitor for OOM issues
- **Speaker similarity**: Validate with reference audio

### TensorBoard Logs:
```bash
tensorboard --logdir ./logs/zero_shot_voice_cloning
# or
tensorboard --logdir ./logs/multi_speaker_voice_cloning
```

## 🎵 Inference After Training

After training completes, you can use the trained LoRA adapter for voice cloning inference:

```python
# Load base model + trained LoRA adapter
model_path = "higgs-audio-v2-generation-3B-base"
adapter_path = "runs/zero_shot_voice_cloning_20250925_123045"

# Use for voice cloning generation
# (Implementation depends on your inference setup)
```

## 🚀 Next Steps

1. **Generate your voice cloning dataset** using the provided script
2. **Start with zero-shot voice cloning** (simpler task)
3. **Monitor training progress** with TensorBoard
4. **Evaluate results** with test samples
5. **Scale to multi-speaker** once comfortable with zero-shot

Good luck with your voice cloning training! 🎤✨