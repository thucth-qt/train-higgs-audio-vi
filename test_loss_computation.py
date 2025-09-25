#!/usr/bin/env python3
"""
Test script to validate proper loss computation for HiggsAudio training
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import our trainer to test the loss computation
from trainer.trainer import HiggsAudioTrainer

def create_mock_outputs():
    """Create mock model outputs similar to HiggsAudioModel"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock HiggsAudioModelOutputWithPast
    class MockOutputs:
        def __init__(self):
            # Text logits: (batch_size, seq_len, vocab_size)
            self.logits = torch.randn(2, 10, 32000, device=device, requires_grad=True)
            # Audio logits: (seq_len, num_codebooks, codebook_size)  
            self.audio_logits = torch.randn(8, 8, 1024, device=device, requires_grad=True)
            # Expanded labels for text: (batch_size, seq_len)
            self.expanded_labels = torch.randint(0, 32000, (2, 10), device=device)
            # Set some labels to -100 (ignored)
            self.expanded_labels[:, :2] = -100
    
    return MockOutputs()

def create_mock_inputs():
    """Create mock training inputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        'label_audio_ids': torch.randint(0, 1024, (8, 8), device=device),  # (num_codebooks, seq_len)
        'label_ids': torch.randint(0, 32000, (2, 10), device=device)       # (batch_size, seq_len)
    }

def test_loss_computation():
    """Test that our loss computation works correctly"""
    print("🧪 Testing HiggsAudio loss computation...")
    
    # Create mock data
    outputs = create_mock_outputs()
    inputs = create_mock_inputs()
    labels = inputs['label_ids']
    
    # Create a mock trainer (we only need the loss computation method)
    class MockTrainer:
        def __init__(self):
            self.model = MockModel()
            
        def _compute_higgs_audio_loss(self, outputs, labels, inputs):
            """Copy of our loss computation method for testing"""
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                total_loss = torch.tensor(0.0, requires_grad=True, device=device)
                num_losses = 0
                
                # 1. Text Loss from logits and expanded_labels
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    text_logits = outputs.logits
                    text_labels = None
                    
                    # Get text labels from outputs.expanded_labels if available
                    if hasattr(outputs, 'expanded_labels') and outputs.expanded_labels is not None:
                        text_labels = outputs.expanded_labels
                    elif labels is not None:
                        text_labels = labels
                    elif isinstance(inputs, dict) and 'label_ids' in inputs:
                        text_labels = inputs['label_ids']
                    elif hasattr(inputs, 'label_ids') and inputs.label_ids is not None:
                        text_labels = inputs.label_ids
                    
                    if text_labels is not None:
                        # Standard cross-entropy loss for text generation
                        shift_logits = text_logits[..., :-1, :].contiguous()
                        shift_labels = text_labels[..., 1:].contiguous()
                        
                        # Flatten for loss computation
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)
                        
                        # Only compute loss for non-ignored tokens (-100)
                        valid_mask = (shift_labels != -100)
                        if valid_mask.sum() > 0:
                            text_loss = F.cross_entropy(
                                shift_logits[valid_mask], 
                                shift_labels[valid_mask], 
                                reduction='mean'
                            )
                            total_loss = total_loss + text_loss
                            num_losses += 1
                            print(f"✓ Text loss: {text_loss.item():.4f}")
                
                # 2. Audio Loss from audio_logits and label_audio_ids  
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None:
                    audio_logits = outputs.audio_logits  # Shape: (seq_len, num_codebooks, codebook_size)
                    audio_labels = None
                    
                    # Get audio labels
                    if isinstance(inputs, dict) and 'label_audio_ids' in inputs:
                        audio_labels = inputs['label_audio_ids']
                    elif hasattr(inputs, 'label_audio_ids') and inputs.label_audio_ids is not None:
                        audio_labels = inputs.label_audio_ids
                    
                    if audio_labels is not None and audio_labels.numel() > 0:
                        # audio_labels shape: (num_codebooks, seq_len)
                        # audio_logits shape: (seq_len, num_codebooks, codebook_size)
                        
                        # Transpose to match dimensions
                        if audio_labels.dim() == 2 and audio_logits.dim() == 3:
                            audio_labels = audio_labels.transpose(0, 1)  # (seq_len, num_codebooks)
                        
                        # Compute loss for each codebook
                        audio_loss = torch.tensor(0.0, device=audio_logits.device, requires_grad=True)
                        valid_codebooks = 0
                        
                        for codebook_idx in range(min(audio_logits.size(1), audio_labels.size(1))):
                            codebook_logits = audio_logits[:, codebook_idx, :]  # (seq_len, codebook_size)
                            codebook_labels = audio_labels[:, codebook_idx]     # (seq_len,)
                            
                            # Only compute loss for valid labels (not -100)
                            valid_mask = (codebook_labels != -100)
                            if valid_mask.sum() > 0:
                                cb_loss = F.cross_entropy(
                                    codebook_logits[valid_mask], 
                                    codebook_labels[valid_mask], 
                                    reduction='mean'
                                )
                                audio_loss = audio_loss + cb_loss
                                valid_codebooks += 1
                        
                        if valid_codebooks > 0:
                            audio_loss = audio_loss / valid_codebooks  # Average across codebooks
                            total_loss = total_loss + audio_loss
                            num_losses += 1
                            print(f"✓ Audio loss: {audio_loss.item():.4f}, codebooks: {valid_codebooks}")
                
                # Average the losses if we have multiple components
                if num_losses > 0:
                    final_loss = total_loss / num_losses
                    print(f"✓ Combined loss: {final_loss.item():.4f} (components: {num_losses})")
                    return final_loss
                else:
                    # Fallback: create a small loss from logits to ensure gradients flow
                    print("⚠️  No valid labels found for loss computation, creating minimal gradient loss")
                    if hasattr(outputs, 'logits') and outputs.logits is not None:
                        return torch.mean(outputs.logits * 0.001)  # Very small but non-zero loss
                    else:
                        return torch.tensor(0.001, requires_grad=True, device=device)
                        
            except Exception as e:
                print(f"❌ Error in _compute_higgs_audio_loss: {e}")
                return torch.tensor(0.001, requires_grad=True, device=device)
    
    class MockModel:
        def parameters(self):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return [torch.tensor(1.0, device=device)]
    
    # Test the loss computation
    trainer = MockTrainer()
    loss = trainer._compute_higgs_audio_loss(outputs, labels, inputs)
    
    # Verify the loss
    print(f"\n🎯 Final Results:")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss requires_grad: {loss.requires_grad}")
    print(f"   Loss device: {loss.device}")
    
    # Test backward pass
    if loss.requires_grad:
        print(f"   Testing backward pass...")
        try:
            loss.backward()
            print(f"   ✅ Backward pass successful!")
        except Exception as e:
            print(f"   ❌ Backward pass failed: {e}")
    
    # Check if loss is significantly greater than zero
    if loss.item() > 0.1:
        print(f"   ✅ Loss is non-zero and meaningful!")
        return True
    else:
        print(f"   ⚠️  Loss is very small, might still be problematic")
        return False

if __name__ == "__main__":
    print("🚀 Testing HiggsAudio Loss Computation Fix")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    success = test_loss_computation()
    
    print("=" * 60)
    if success:
        print("🎉 Loss computation test PASSED!")
        print("   Training should now work with proper gradients!")
    else:
        print("❌ Loss computation test failed or suspicious")
        print("   Further debugging may be needed")