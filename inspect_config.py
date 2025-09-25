#!/usr/bin/env python3
"""
Simple config file inspector for debugging model loading issues
Run this on the server to examine the config.json file in detail
"""

import json
import sys
from pathlib import Path

def inspect_config(model_path):
    """Inspect the config.json file for potential issues"""
    config_file = Path(model_path) / "config.json"
    
    print(f"Inspecting config file: {config_file}")
    print("=" * 60)
    
    if not config_file.exists():
        print(f"❌ Config file does not exist: {config_file}")
        return False
    
    try:
        # Read raw content
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        print(f"✓ File size: {len(raw_content)} characters")
        print(f"✓ First 200 chars: {raw_content[:200]}")
        print(f"✓ Last 200 chars: {raw_content[-200:]}")
        
        # Parse JSON
        config_data = json.loads(raw_content)
        print(f"✓ Valid JSON with {len(config_data)} top-level keys")
        
        # Show all keys and their types
        print("\nTop-level keys and types:")
        for key, value in sorted(config_data.items()):
            value_type = type(value).__name__
            if isinstance(value, dict):
                print(f"  {key}: {value_type} ({len(value)} keys)")
            elif isinstance(value, list):
                print(f"  {key}: {value_type} ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 50:
                print(f"  {key}: {value_type} (length {len(value)})")
            else:
                print(f"  {key}: {value_type} = {value}")
        
        # Check for None values that might cause issues
        print("\nChecking for problematic None values:")
        none_count = 0
        def check_none_recursive(obj, path=""):
            nonlocal none_count
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    if v is None:
                        print(f"  {new_path}: None")
                        none_count += 1
                    else:
                        check_none_recursive(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    if item is None:
                        print(f"  {new_path}: None")
                        none_count += 1
                    else:
                        check_none_recursive(item, new_path)
        
        check_none_recursive(config_data)
        if none_count == 0:
            print("  No None values found")
        else:
            print(f"  Found {none_count} None values")
        
        # Check specific important fields
        print("\nImportant fields:")
        important_fields = [
            "model_type", "architectures", "_name_or_path",
            "text_config", "audio_encoder_config", 
            "audio_num_codebooks", "audio_codebook_size"
        ]
        
        for field in important_fields:
            if field in config_data:
                value = config_data[field]
                if isinstance(value, dict):
                    print(f"  {field}: dict with keys {list(value.keys())[:5]}...")
                else:
                    print(f"  {field}: {value}")
            else:
                print(f"  {field}: MISSING")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base"
    inspect_config(model_path)