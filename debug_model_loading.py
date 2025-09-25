#!/usr/bin/env python3
"""Debug script to identify model loading issue"""

import sys
import traceback
from pathlib import Path

def test_config_loading():
    """Test model config loading with detailed error reporting"""
    model_path = "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base"
    
    print(f"Testing config loading from: {model_path}")
    
    # Check if path exists
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Check config file
    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        print(f"❌ Config file does not exist: {config_file}")
        return False
    
    print(f"✓ Config file exists: {config_file}")
    
    # Try to read config file manually
    try:
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"✓ Config file is valid JSON with keys: {list(config_data.keys())}")
        
        # Print some key fields
        if 'model_type' in config_data:
            print(f"  - Model type: {config_data['model_type']}")
        if 'architectures' in config_data:
            print(f"  - Architectures: {config_data['architectures']}")
            
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        traceback.print_exc()
        return False
    
    # Try importing the config class
    try:
        sys.path.append('/Users/thuc.tran/thucth/train-higgs-audio-vi')
        from boson_multimodal.model.higgs_audio import HiggsAudioConfig
        print("✓ HiggsAudioConfig imported successfully")
    except Exception as e:
        print(f"❌ Error importing HiggsAudioConfig: {e}")
        traceback.print_exc()
        return False
    
    # Try loading the config
    try:
        print("Loading config with from_pretrained...")
        config = HiggsAudioConfig.from_pretrained(model_path)
        print(f"✓ Config loaded successfully: {type(config)}")
        print(f"  - Model type: {config.model_type}")
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_loading()