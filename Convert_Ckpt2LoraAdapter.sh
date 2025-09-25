python3 /root/data/higgs/train-higgs-audio-vi/trainer/convert_checkpoint_to_lora.py \
--base_model_path /root/data/higgs/weights/higgs-audio-v2-generation-3B-base \
--checkpoint_path /root/data/higgs/train-higgs-audio-vi/runs/zero_shot_voice_cloning_memory_optimized_20250925_044303/checkpoint-1000 \
--output_path /root/data/higgs/train-higgs-audio-vi/runs/zero_shot_voice_cloning_memory_optimized_20250925_044303_lora_step1000