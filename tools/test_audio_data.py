# test_audio_data.py
import torch
from pathlib import Path
import glob

def test_audio_tokenizer():
    # 导入你的audio tokenizer
    try:
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_tokenizer = load_higgs_audio_tokenizer("/root/data/higgs/weights", device=device)
        
        # Test a few Vietnamese audio files
        data_dir = "/root/data/higgs/train-higgs-audio-vi/vietnamese_training_data"
        audio_files = list(Path(data_dir).glob("*.wav")) + list(Path(data_dir).glob("*.mp3"))
        
        print(f"Found {len(audio_files)} audio files")
        
        for i, audio_file in enumerate(audio_files[:5]):  # 只测试前5个
            print(f"\n--- Testing file {i+1}: {audio_file.name} ---")
            try:
                tokens = audio_tokenizer.encode(str(audio_file))
                if tokens is not None:
                    if isinstance(tokens, list):
                        print(f"Length: {len(tokens)}")
                        print(f"Type: {type(tokens[0]) if tokens else 'empty'}")
                        if tokens:
                            print(f"Min value: {min(tokens)}")
                            print(f"Max value: {max(tokens)}")
                            print(f"Sample values: {tokens[:5]}")
                    else:
                        print(f"Shape: {tokens.shape}")
                        print(f"Dtype: {tokens.dtype}")
                        print(f"Min value: {tokens.min().item()}")
                        print(f"Max value: {tokens.max().item()}")
                        print(f"Sample values: {tokens[:, :5] if tokens.shape[1] > 5 else tokens}")
                else:
                    print("Returned None")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                
    except ImportError as e:
        print(f"Cannot import audio tokenizer: {e}")

if __name__ == "__main__":
    test_audio_tokenizer()