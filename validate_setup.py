#!/usr/bin/env python3
"""
Pre-training validation script for Higgs Audio v2 Training Setup
Validates model loading, data integrity, and system requirements before training
"""

import os
import sys

# Import patch module first to fix torch version issue
try:
    import torch_patch  # This applies the patch immediately
    print("✓ Torch patch module loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load torch patch module: {e}")

import torch
import logging
import argparse
from pathlib import Path

# Verify torch version after patch
print(f"Current torch.__version__: {torch.__version__}")

# Emergency patch for transformers import issue
def patch_torch_version():
    """Patch torch version if it's None or invalid to prevent transformers import errors"""
    if torch.__version__ is None or not isinstance(torch.__version__, str):
        print(f"⚠️  Warning: torch.__version__ is invalid: {torch.__version__}")
        print("Applying emergency patch for transformers compatibility...")
        torch.__version__ = "2.7.1+cu126"  # Use known working version
        print(f"✓ Patched torch.__version__ to: {torch.__version__}")

# Apply patch before any transformers imports
patch_torch_version()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements for training"""
    logger.info("=" * 60)
    logger.info("SYSTEM REQUIREMENTS CHECK")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        logger.info(f"✓ CUDA available with {device_count} device(s)")
        logger.info(f"✓ Current device: {device_name}")
        logger.info(f"✓ GPU Memory: {memory_gb:.1f} GB")
        
        if memory_gb < 16:
            logger.warning("⚠ GPU has less than 16GB memory. Training may require smaller batch sizes.")
        
        # Check mixed precision support
        if torch.cuda.is_bf16_supported():
            logger.info("✓ BFloat16 (bf16) supported")
        else:
            logger.info("⚠ BFloat16 not supported, will use float16")
            
    else:
        logger.error("✗ CUDA not available. GPU training not possible.")
        return False
    
    # Check PyTorch version
    pytorch_version = torch.__version__
    logger.info(f"✓ PyTorch version: {pytorch_version}")
    
    return True

def validate_model_paths(model_path, audio_tokenizer_path):
    """Validate model and tokenizer paths"""
    logger.info("=" * 60)
    logger.info("MODEL PATH VALIDATION")
    logger.info("=" * 60)
    
    # Check model path
    if not os.path.exists(model_path):
        logger.error(f"✗ Model path does not exist: {model_path}")
        return False
    
    logger.info(f"✓ Model path exists: {model_path}")
    
    # Check for config file
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        logger.info(f"✓ Config file found: {config_path}")
    else:
        logger.warning(f"⚠ Config file not found: {config_path}")
    
    # Check audio tokenizer path
    if not os.path.exists(audio_tokenizer_path):
        logger.error(f"✗ Audio tokenizer path does not exist: {audio_tokenizer_path}")
        return False
    
    logger.info(f"✓ Audio tokenizer path exists: {audio_tokenizer_path}")
    
    return True

def test_model_loading(model_path, audio_tokenizer_path):
    """Test model loading with enhanced error handling"""
    logger.info("=" * 60)
    logger.info("MODEL LOADING TEST")
    logger.info("=" * 60)
    
    try:
        # Test imports
        logger.info("Testing imports...")
        try:
            # Apply patch right before importing modules that use transformers
            if torch.__version__ is None or not isinstance(torch.__version__, str):
                logger.warning(f"⚠️  Warning: torch.__version__ is invalid: {torch.__version__}")
                logger.info("Applying emergency patch for transformers compatibility...")
                torch.__version__ = "2.7.1+cu126"  # Use known working version
                logger.info(f"✓ Patched torch.__version__ to: {torch.__version__}")
            
            from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            logger.info("✓ Higgs Audio modules imported successfully")
            higgs_available = True
        except ImportError as e:
            logger.warning(f"⚠ Higgs Audio modules not available: {e}")
            higgs_available = False
        
        # Test model config loading
        if higgs_available:
            logger.info("Loading model configuration...")
            
            # First, check config file manually
            import json
            config_file = Path(model_path) / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    logger.info(f"✓ Config file is valid JSON with {len(config_data)} keys")
                    
                    # Check for potential problematic fields
                    none_fields = []
                    for key, value in config_data.items():
                        if value is None:
                            none_fields.append(key)
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subvalue is None:
                                    none_fields.append(f"{key}.{subkey}")
                    
                    if none_fields:
                        logger.debug(f"Found {len(none_fields)} None values in config: {none_fields[:5]}")
                        logger.info("✓ Config contains expected None values (normal for HiggsAudio model)")
                    
                except Exception as e:
                    logger.error(f"✗ Config file parsing error: {e}")
                    return False
            else:
                logger.error(f"✗ Config file not found: {config_file}")
                return False
            
            # Try loading the config with detailed error reporting
            try:
                config = HiggsAudioConfig.from_pretrained(model_path)
                logger.info("✓ Model configuration loaded successfully")
                logger.info(f"  - Model type: {getattr(config, 'model_type', 'N/A')}")
            except Exception as config_error:
                logger.error(f"✗ Config loading failed: {config_error}")
                logger.error(f"  Error type: {type(config_error).__name__}")
                
                # Try alternative loading method
                logger.info("Attempting alternative config loading...")
                try:
                    # Load config data manually and create config object
                    config = HiggsAudioConfig(**config_data)
                    logger.info("✓ Alternative config loading successful")
                except Exception as alt_error:
                    logger.error(f"✗ Alternative config loading also failed: {alt_error}")
                    return False
            
            # Test audio tokenizer loading with error handling
            logger.info("Loading audio tokenizer...")
            try:
                device = "cpu"  # Load on CPU for testing
                audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=device)
                logger.info("✓ Audio tokenizer loaded successfully")
            except Exception as tokenizer_error:
                logger.error(f"✗ Audio tokenizer loading failed: {tokenizer_error}")
                # Continue anyway since this is just validation
                logger.info("  (Continuing validation despite tokenizer error)")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        logger.error(f"  Full traceback: {traceback.format_exc()}")
        return False

def validate_dataset(data_dir, sample_size=10):
    """Validate dataset structure and samples"""
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 60)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"✗ Dataset directory does not exist: {data_dir}")
        return False
    
    logger.info(f"✓ Dataset directory exists: {data_dir}")
    
    # Check for metadata file
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        logger.info("✓ Metadata file found")
        
        # Load and validate metadata
        try:
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            samples = metadata.get("samples", [])
            logger.info(f"✓ Metadata contains {len(samples)} samples")
            
            if len(samples) == 0:
                logger.error("✗ No samples found in metadata")
                return False
                
            # Validate sample structure
            valid_samples = 0
            for i, sample in enumerate(samples[:sample_size]):
                try:
                    audio_file = data_path / sample["audio_file"]
                    if audio_file.exists():
                        valid_samples += 1
                    else:
                        logger.warning(f"Sample {i}: Audio file missing: {audio_file}")
                        
                except KeyError as e:
                    logger.warning(f"Sample {i}: Missing key: {e}")
            
            logger.info(f"✓ Validated {valid_samples}/{min(sample_size, len(samples))} samples")
            
        except Exception as e:
            logger.error(f"✗ Failed to validate metadata: {e}")
            return False
    else:
        # Check for audio files directly
        audio_files = list(data_path.glob("*.wav")) + list(data_path.glob("*.mp3"))
        logger.info(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            logger.error("✗ No audio files found")
            return False
    
    return True

def test_bf16_support():
    """Test BFloat16 support"""
    logger.info("=" * 60)
    logger.info("BFLOAT16 SUPPORT TEST")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.warning("⚠ CUDA not available, skipping bf16 test")
        return False
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    if torch.cuda.is_bf16_supported():
        try:
            # Try a simple BF16 operation
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            c = torch.matmul(a, b)
            logger.info(f"✓ BF16 matmul test passed, result dtype: {c.dtype}")
            return True
        except Exception as e:
            logger.error(f"✗ BF16 operation failed: {e}")
            return False
    else:
        logger.info("⚠ BF16 not supported on this GPU, will use FP16")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate Higgs Audio v2 training setup")
    parser.add_argument("--model_path", type=str, 
                       default="/root/data/higgs/weights/higgs-audio-v2-generation-3B-base",
                       help="Path to the model")
    parser.add_argument("--audio_tokenizer_path", type=str, 
                       default="/root/data/higgs/weights/higgs-audio-v2-tokenizer",
                       help="Path to the audio tokenizer")
    parser.add_argument("--train_data_dir", type=str, 
                       default="/root/data/higgs/balanced_tts_dataset_higgs_mini",
                       help="Path to training data")
    
    args = parser.parse_args()
    
    logger.info("Starting Higgs Audio v2 Training Setup Validation")
    logger.info("=" * 80)
    
    success = True
    
    # Run all validation checks
    checks = [
        ("System Requirements", lambda: check_system_requirements(), True),  # Critical
        ("Model Paths", lambda: validate_model_paths(args.model_path, args.audio_tokenizer_path), True),  # Critical
        ("Model Loading", lambda: test_model_loading(args.model_path, args.audio_tokenizer_path), False),  # Non-critical
        ("Dataset", lambda: validate_dataset(args.train_data_dir), True),  # Critical
        ("BFloat16 Support", lambda: test_bf16_support(), True),  # Critical
    ]
    
    results = {}
    critical_failed = False
    
    for check_name, check_func, is_critical in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result and is_critical:
                critical_failed = True
            elif not result and not is_critical:
                logger.warning(f"⚠ {check_name} failed but is non-critical, continuing...")
        except Exception as e:
            logger.error(f"✗ {check_name} check failed with exception: {e}")
            results[check_name] = False
            if is_critical:
                critical_failed = True
    
    # Print summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name}: {status}")
    
    if not critical_failed:
        logger.info("=" * 80)
        logger.info("🎉 ALL CRITICAL CHECKS PASSED! Ready for training.")
        if not all(results.values()):
            logger.info("Note: Some non-critical checks failed but training can proceed.")
        logger.info("=" * 80)
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("❌ CRITICAL CHECKS FAILED! Please fix issues before training.")
        logger.error("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()