# Set the model path as an environment variable
# export MODEL_PATH="/root/data/higgs/weights/higgs-audio-v2-generation-3B-base"
export MODEL_PATH="/root/data/higgs/train-higgs-audio-vi/runs/2_output_vn_lora_mini_merged"

python3 examples/generation.py \
--model_path "$MODEL_PATH" \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation-en.wav

python3 examples/generation.py \
--model_path "$MODEL_PATH" \
--transcript "Vui lòng tạo ra một đoạn âm thanh để mô tả hôm nay là ngày 23 tháng 9 năm 2025" \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation-vi.wav