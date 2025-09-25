#!/usr/bin/env python3
"""Debug script for server-side model loading issue"""

import sys
import json
import traceback
from pathlib import Path

def debug_config_loading():
    """Debug the config loading process step by step"""
    model_path = "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base"
    config_file = Path(model_path) / "config.json"
    
    print("=" * 60)
    print("DEBUGGING MODEL CONFIG LOADING")
    print("=" * 60)
    
    # Step 1: Check paths
    print(f"1. Checking model path: {model_path}")
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist")
        return False
    print(f"✓ Model path exists")
    
    print(f"2. Checking config file: {config_file}")
    if not config_file.exists():
        print(f"❌ Config file does not exist")
        return False
    print(f"✓ Config file exists")
    
    # Step 2: Read and analyze config file
    print(f"3. Reading config file contents...")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        print(f"✓ Raw content read successfully ({len(raw_content)} chars)")
        print(f"First 200 chars: {raw_content[:200]}...")
        
        # Parse as JSON
        config_data = json.loads(raw_content)
        print(f"✓ Valid JSON with {len(config_data)} keys")
        print(f"Keys: {list(config_data.keys())}")
        
        # Check key fields
        important_keys = ['model_type', 'architectures', 'text_config', 'audio_encoder_config']
        for key in important_keys:
            if key in config_data:
                print(f"  - {key}: {type(config_data[key])}")
                if isinstance(config_data[key], dict):
                    print(f"    Dict keys: {list(config_data[key].keys()) if len(config_data[key]) < 20 else f'{len(config_data[key])} keys'}")
                elif isinstance(config_data[key], (str, int, float, bool)):
                    print(f"    Value: {config_data[key]}")
                elif isinstance(config_data[key], list):
                    print(f"    List length: {len(config_data[key])}")
            else:
                print(f"  - {key}: MISSING")
                
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Test imports
    print(f"4. Testing imports...")
    try:
        sys.path.append('/root/data/higgs/train-higgs-audio-vi')
        from boson_multimodal.model.higgs_audio import HiggsAudioConfig
        print(f"✓ HiggsAudioConfig imported successfully")
    except Exception as e:
        print(f"❌ Error importing HiggsAudioConfig: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Test config loading with detailed error handling
    print(f"5. Testing HiggsAudioConfig.from_pretrained...")
    try:
        # First try to create a config manually
        print("  5a. Creating config from dict...")
        test_config = HiggsAudioConfig(**config_data)
        print(f"  ✓ Manual config creation successful")
        
        # Then try from_pretrained
        print("  5b. Using from_pretrained...")
        config = HiggsAudioConfig.from_pretrained(model_path)
        print(f"  ✓ from_pretrained successful: {type(config)}")
        print(f"  - Model type: {getattr(config, 'model_type', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Error in config loading: {e}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        
        # Try to get more specific info
        if "string or bytes-like object" in str(e):
            print("\n🔍 SPECIFIC ANALYSIS for 'string or bytes-like object' error:")
            print("This error typically occurs when:")
            print("1. A regex pattern expects a string but gets None or other type")
            print("2. JSON parsing issues with non-string values")
            print("3. File path handling issues")
            
            # Check for problematic fields
            problematic_fields = []
            for key, value in config_data.items():
                if value is None:
                    problematic_fields.append(f"{key}: None")
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subvalue is None:
                            problematic_fields.append(f"{key}.{subkey}: None")
            
            if problematic_fields:
                print(f"Found {len(problematic_fields)} None values that might cause issues:")
                for field in problematic_fields[:10]:  # Show first 10
                    print(f"  - {field}")
            else:
                print("No obvious None values found")
        
        return False

if __name__ == "__main__":
    debug_config_loading()