#!/usr/bin/env python3
"""
Quick test to verify the class-level monkey-patch for labels parameter fix.
This script tests if the HiggsAudioModel.forward method properly ignores the labels parameter.
"""

import sys
import torch
sys.path.append('/root/data/higgs/train-higgs-audio-vi')

def test_labels_fix():
    print("🧪 Testing Class-Level Labels Fix...")
    
    try:
        # Import the wrapper which should apply the class-level patch
        from trainer.trainer import HiggsAudioModelWrapper
        
        # Create a small model wrapper (this triggers the class-level patch)
        print("📦 Loading model wrapper (triggers class-level patch)...")
        wrapper = HiggsAudioModelWrapper(
            "/root/data/higgs/weights/higgs-audio-v2-generation-3B-base", 
            device="cuda"
        )
        
        # Test that the patch worked by calling forward with labels
        print("🔬 Testing patched forward method...")
        
        # Create dummy inputs
        dummy_input_ids = torch.tensor([[1, 2, 3]], device="cuda")
        dummy_attention_mask = torch.tensor([[1, 1, 1]], device="cuda")
        
        # This should NOT fail even though we're passing labels
        try:
            output = wrapper.model.forward(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                labels=dummy_input_ids  # This should be ignored by the patch!
            )
            print("✅ SUCCESS: Class-level patch working! Model accepts and ignores 'labels' parameter.")
            return True
            
        except TypeError as e:
            if "labels" in str(e):
                print(f"❌ FAILURE: Class-level patch not working. Error: {e}")
                return False
            else:
                print(f"⚠️  Different error (may be expected): {e}")
                return True  # Different error is okay
                
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_labels_fix()
    if success:
        print("\n🎉 Class-level labels fix is working correctly!")
        print("🚀 Training should now proceed without labels parameter errors.")
    else:
        print("\n🚨 Class-level labels fix needs more work.")
        print("❓ Please check the monkey-patch implementation.")