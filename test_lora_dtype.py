from safetensors.torch import safe_open

def print_safetensors_dtypes(filepath):
    with safe_open(filepath, framework="pt") as f:
        print(f"File: {filepath}")
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"{key}: {tensor.dtype}")

# LoRA adapter
print_safetensors_dtypes("/root/data/higgs/train-higgs-audio-vi/output_vn_lora_exp2/lora_adapters/adapter_model.safetensors")

# Base model shards
# for i in range(1, 4):
print_safetensors_dtypes(f"/root/data/higgs/weights/higgs-audio-v2-generation-3B-base/model-00001-of-00003.safetensors")
print_safetensors_dtypes(f"/root/data/higgs/train-higgs-audio-vi/merged_models/vietnamese_lora_merged/model-00001-of-00003.safetensors")
