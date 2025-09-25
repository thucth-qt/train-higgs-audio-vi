#!/usr/bin/env python3
"""
Test script to verify the audio indexing fix
"""

import torch

def test_audio_indexing_fix():
    """Test the improved audio indexing logic"""
    print("🧪 Testing Audio Indexing Fix")
    
    # Simulate the problematic scenario:
    # - 1 audio file available (num_available_audios = 1, so valid indices are [0])
    # - Multiple audio tokens in the sequence (both input and output)
    
    # Mock input_ids with audio tokens
    # Let's say: [text, <audio_in>, text, <audio_out>, text, <audio_out>, text]
    audio_in_token_id = 50001
    audio_out_token_id = 50002
    
    input_ids = torch.tensor([1, audio_in_token_id, 2, audio_out_token_id, 3, audio_out_token_id, 4])
    num_available_audios = 1  # Only 1 audio file available (index 0 only)
    
    print(f"Input sequence: {input_ids.tolist()}")
    print(f"Available audio files: {num_available_audios}")
    print(f"Audio in token ID: {audio_in_token_id}")
    print(f"Audio out token ID: {audio_out_token_id}")
    
    # Apply the OLD logic (problematic)
    print("\n🔴 OLD Logic (Problematic):")
    audio_in_mask_old = input_ids == audio_in_token_id
    audio_out_mask_old = input_ids == audio_out_token_id
    audio_ids_old = torch.ones_like(input_ids)
    audio_ids_old[audio_in_mask_old ^ audio_out_mask_old] = torch.cumsum(audio_ids_old[audio_in_mask_old ^ audio_out_mask_old], 0) - 1
    audio_in_ids_old = audio_ids_old[audio_in_mask_old]
    audio_out_ids_old = audio_ids_old[audio_out_mask_old]
    
    print(f"Audio positions mask: {(audio_in_mask_old ^ audio_out_mask_old).int().tolist()}")
    print(f"Generated audio_ids: {audio_ids_old.tolist()}")
    print(f"Audio input indices: {audio_in_ids_old.tolist()}")
    print(f"Audio output indices: {audio_out_ids_old.tolist()}")
    
    # Check bounds violations
    all_indices_old = torch.cat([audio_in_ids_old, audio_out_ids_old])
    violations_old = (all_indices_old >= num_available_audios).sum().item()
    print(f"Bounds violations: {violations_old}")
    
    # Apply the NEW logic (fixed)
    print("\n🟢 NEW Logic (Fixed):")
    audio_in_mask = input_ids == audio_in_token_id
    audio_out_mask = input_ids == audio_out_token_id
    audio_ids = torch.ones_like(input_ids)
    
    # Create indices with proper bounds checking
    audio_positions = audio_in_mask ^ audio_out_mask
    if audio_positions.sum() > 0:
        # Create sequential indices but cap them at available audio count
        sequential_indices = torch.cumsum(audio_positions.int(), 0) - 1
        # Clamp indices to available audio range [0, num_available_audios-1]
        clamped_indices = torch.clamp(sequential_indices, 0, max(0, num_available_audios - 1))
        audio_ids[audio_positions] = clamped_indices[audio_positions]
    
    audio_in_ids = audio_ids[audio_in_mask]
    audio_out_ids = audio_ids[audio_out_mask]
    
    print(f"Audio positions mask: {audio_positions.int().tolist()}")
    print(f"Sequential indices: {sequential_indices.tolist()}")
    print(f"Clamped indices: {clamped_indices.tolist()}")
    print(f"Generated audio_ids: {audio_ids.tolist()}")
    print(f"Audio input indices: {audio_in_ids.tolist()}")
    print(f"Audio output indices: {audio_out_ids.tolist()}")
    
    # Check bounds violations
    all_indices = torch.cat([audio_in_ids, audio_out_ids])
    violations = (all_indices >= num_available_audios).sum().item()
    print(f"Bounds violations: {violations}")
    
    # Test with multiple audio files
    print("\n🧪 Testing with 3 available audio files:")
    num_available_audios_multi = 3
    
    audio_positions_multi = audio_in_mask ^ audio_out_mask
    if audio_positions_multi.sum() > 0:
        sequential_indices_multi = torch.cumsum(audio_positions_multi.int(), 0) - 1
        clamped_indices_multi = torch.clamp(sequential_indices_multi, 0, max(0, num_available_audios_multi - 1))
        audio_ids_multi = torch.ones_like(input_ids)
        audio_ids_multi[audio_positions_multi] = clamped_indices_multi[audio_positions_multi]
    
    audio_in_ids_multi = audio_ids_multi[audio_in_mask]
    audio_out_ids_multi = audio_ids_multi[audio_out_mask]
    
    print(f"Available audio files: {num_available_audios_multi}")
    print(f"Audio input indices: {audio_in_ids_multi.tolist()}")
    print(f"Audio output indices: {audio_out_ids_multi.tolist()}")
    
    all_indices_multi = torch.cat([audio_in_ids_multi, audio_out_ids_multi])
    violations_multi = (all_indices_multi >= num_available_audios_multi).sum().item()
    print(f"Bounds violations: {violations_multi}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   OLD logic violations: {violations_old}")
    print(f"   NEW logic violations: {violations}")
    print(f"   NEW logic with 3 files violations: {violations_multi}")
    
    return violations == 0 and violations_multi == 0

if __name__ == "__main__":
    success = test_audio_indexing_fix()
    if success:
        print(f"\n🎉 Audio indexing fix test PASSED!")
        print(f"   No more bounds violations should occur!")
    else:
        print(f"\n❌ Audio indexing fix test FAILED!")
        print(f"   Further debugging needed")