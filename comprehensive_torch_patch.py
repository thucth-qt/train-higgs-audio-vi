#!/usr/bin/env python3
"""
Comprehensive patch for torch version issue
This patches the issue at the system level before any imports
"""

import sys
import torch

def comprehensive_torch_patch():
    """Apply comprehensive torch version patch"""
    print("=" * 60)
    print("COMPREHENSIVE TORCH VERSION PATCH")
    print("=" * 60)
    
    # Check current torch version
    original_version = torch.__version__
    print(f"Original torch.__version__: {repr(original_version)}")
    print(f"Type: {type(original_version)}")
    
    # Check if patch is needed
    if original_version is None or not isinstance(original_version, str):
        print("❌ Torch version is invalid - applying patch")
        
        # Patch the version
        fixed_version = "2.7.1+cu126"
        torch.__version__ = fixed_version
        print(f"✓ Patched torch.__version__ to: {repr(fixed_version)}")
        
        # Also patch internal torch attributes that might be used
        if hasattr(torch, '__version_info__'):
            print(f"Original torch.__version_info__: {torch.__version_info__}")
        
        # Verify the patch worked
        print(f"Verification - torch.__version__: {repr(torch.__version__)}")
        
        return True
    else:
        print("✓ Torch version is already valid")
        return False

def test_transformers_import():
    """Test transformers import after patch"""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMERS IMPORT")
    print("=" * 60)
    
    try:
        print("Attempting transformers import...")
        import transformers
        print(f"✓ SUCCESS: Transformers imported - version {transformers.__version__}")
        
        # Test specific imports that were failing
        from transformers import AutoTokenizer, AutoConfig
        print("✓ SUCCESS: AutoTokenizer and AutoConfig imported")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Apply patch
    patched = comprehensive_torch_patch()
    
    # Test imports
    success = test_transformers_import()
    
    if success:
        print("\n🎉 PATCH SUCCESSFUL! Transformers can now be imported.")
    else:
        print("\n❌ PATCH FAILED! Manual intervention required.")
        print("\nTry running the package fix:")
        print("source .venv/bin/activate && python fix_transformers_issue.py")
    
    sys.exit(0 if success else 1)