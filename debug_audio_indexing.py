#!/usr/bin/env python3

# Debug script to understand the audio index assignment issue

import torch

# Simulate the problematic scenario
def debug_audio_indexing():
    # Let's say we have a sequence with audio tokens
    # audio_in_token_id = 32000, audio_out_token_id = 32001
    audio_in_token_id = 32000
    audio_out_token_id = 32001
    
    # Example input_ids with multiple audio tokens but only 1 audio available
    input_ids = torch.tensor([1, 2, 32000, 3, 4, 32001, 5, 6])  # 1 audio_in + 1 audio_out
    print(f"Input IDs: {input_ids}")
    
    # Create masks
    audio_in_mask = input_ids == audio_in_token_id
    audio_out_mask = input_ids == audio_out_token_id
    
    print(f"Audio in mask:  {audio_in_mask}")
    print(f"Audio out mask: {audio_out_mask}")
    
    # The problematic logic
    audio_ids = torch.ones_like(input_ids)
    print(f"Initial audio_ids: {audio_ids}")
    
    combined_mask = audio_in_mask | audio_out_mask  # Use OR instead of XOR for clarity
    print(f"Combined mask: {combined_mask}")
    
    # This is where the issue is - cumsum assigns sequential indices
    audio_ids[combined_mask] = torch.cumsum(audio_ids[combined_mask], 0) - 1
    print(f"After cumsum audio_ids: {audio_ids}")
    
    audio_in_ids = audio_ids[audio_in_mask]
    audio_out_ids = audio_ids[audio_out_mask]
    
    print(f"Audio in IDs: {audio_in_ids}")
    print(f"Audio out IDs: {audio_out_ids}")
    print(f"Available audios: 1 (only index 0 is valid)")
    print(f"Problem: audio_out_ids contains {audio_out_ids.item()}, but max valid index is 0")

if __name__ == "__main__":
    debug_audio_indexing()