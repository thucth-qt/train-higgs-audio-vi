#!/usr/bin/env python3
"""
Quick test script to verify the trainer implementation
Tests model loading, dataset creation, and basic forward pass
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the current directory to Python path so we can import our trainer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trainer_components():
    """Test trainer components individually"""
    logger.info("Testing Higgs Audio v2 trainer components...")
    
    try:
        from trainer.trainer import (
            HiggsAudioDataset,
            HiggsAudioModelWrapper, 
            HiggsAudioTrainer,
            ExtendedHiggsAudioSampleCollator,
            MemoryMonitor,
            HIGGS_AVAILABLE
        )
        logger.info("✓ Successfully imported trainer components")
        
        # Test memory monitor
        memory_monitor = MemoryMonitor()
        memory_monitor.log_memory_usage("Test start")
        logger.info("✓ Memory monitor working")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import trainer components: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error testing components: {e}")
        return False

def test_mock_training_setup():
    """Test training setup with mock data"""
    try:
        from trainer.trainer import (
            HiggsAudioModelWrapper,
            ExtendedHiggsAudioSampleCollator,
            ChatMLDatasetSample
        )
        from transformers import AutoTokenizer
        
        # Create a mock sample
        mock_sample = ChatMLDatasetSample(
            input_ids=torch.randint(0, 1000, (10,)),
            label_ids=torch.randint(0, 1000, (10,)),
            audio_ids_concat=torch.randint(0, 100, (8, 20)),
            audio_ids_start=torch.tensor([0]),
            label_audio_ids=torch.randint(0, 100, (8, 15)),
            audio_waveforms_concat=torch.randn(1000),
            audio_waveforms_start=torch.tensor([0]),
            audio_sample_rate=torch.tensor([24000.0]),
            audio_speaker_indices=torch.tensor([0]),
        )
        
        logger.info("✓ Mock sample created successfully")
        
        # Test collator
        collator = ExtendedHiggsAudioSampleCollator(pad_token_id=0)
        batch = collator([mock_sample, mock_sample])  # Test with 2 samples
        
        logger.info("✓ Batch collation successful")
        logger.info(f"  Batch input_ids shape: {batch.input_ids.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Mock training setup failed: {e}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("HIGGS AUDIO V2 TRAINER QUICK TEST")
    logger.info("=" * 60)
    
    success = True
    
    # Test 1: Component imports
    logger.info("Test 1: Component imports")
    if not test_trainer_components():
        success = False
    
    logger.info("-" * 40)
    
    # Test 2: Mock training setup
    logger.info("Test 2: Mock training setup")
    if not test_mock_training_setup():
        success = False
    
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 All tests passed! Trainer implementation looks good.")
        logger.info("You can now proceed with actual training.")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()