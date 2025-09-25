#!/usr/bin/env python3
"""
Debug the exact torch version issue in transformers
This will help identify what exact variable is causing the problem
"""

import torch
import sys

def debug_torch_version():
    """Debug all torch version related attributes"""
    print("=" * 60)
    print("TORCH VERSION DEBUG")
    print("=" * 60)
    
    # Check all version-related attributes
    version_attrs = ['__version__', '_version', '__version_info__']
    
    for attr in version_attrs:
        if hasattr(torch, attr):
            value = getattr(torch, attr)
            print(f"torch.{attr}: {repr(value)} (type: {type(value)})")
        else:
            print(f"torch.{attr}: NOT FOUND")
    
    # Check what transformers is trying to access
    print("\nChecking what transformers might be looking for...")
    
    # Try to find _torch_version that transformers is trying to parse
    try:
        # This is what transformers/utils/import_utils.py is trying to access
        if hasattr(torch, '__version__'):
            _torch_version = torch.__version__
        else:
            _torch_version = None
            
        print(f"_torch_version (what transformers tries to parse): {repr(_torch_version)}")
        
        if _torch_version is None:
            print("❌ This is the problem! _torch_version is None")
            return False
        elif not isinstance(_torch_version, str):
            print(f"❌ This is the problem! _torch_version is not a string: {type(_torch_version)}")
            return False
        else:
            print("✓ _torch_version looks valid for parsing")
            
            # Test the actual parsing that's failing
            print("\nTesting version parsing...")
            from packaging import version
            try:
                parsed = version.parse(_torch_version)
                print(f"✓ Version parsing successful: {parsed}")
                return True
            except Exception as e:
                print(f"❌ Version parsing failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error accessing torch version: {e}")
        return False

def apply_comprehensive_fix():
    """Apply comprehensive fix for all torch version issues"""
    print("\n" + "=" * 60)
    print("APPLYING COMPREHENSIVE FIX")
    print("=" * 60)
    
    # Fix all possible torch version attributes
    fixed_version = "2.7.1+cu126"
    
    # Fix __version__ 
    if torch.__version__ is None or not isinstance(torch.__version__, str):
        print(f"Fixing torch.__version__ from {repr(torch.__version__)} to {repr(fixed_version)}")
        torch.__version__ = fixed_version
    
    # Fix other version attributes if they exist
    version_attrs = ['_version']
    for attr in version_attrs:
        if hasattr(torch, attr):
            old_value = getattr(torch, attr)
            if old_value is None or not isinstance(old_value, str):
                print(f"Fixing torch.{attr} from {repr(old_value)} to {repr(fixed_version)}")
                setattr(torch, attr, fixed_version)
    
    print("✓ All torch version attributes fixed")

def test_transformers_import():
    """Test transformers import after fix"""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMERS IMPORT")
    print("=" * 60)
    
    try:
        print("Attempting transformers import...")
        import transformers
        print(f"✓ SUCCESS! Transformers imported: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Debug current state
    is_valid = debug_torch_version()
    
    if not is_valid:
        # Apply comprehensive fix
        apply_comprehensive_fix()
        
        # Re-test after fix
        print("\nRe-testing after fix...")
        is_valid = debug_torch_version()
    
    # Test transformers import
    if is_valid:
        success = test_transformers_import()
    else:
        print("❌ Could not fix torch version issue")
        success = False
    
    sys.exit(0 if success else 1)