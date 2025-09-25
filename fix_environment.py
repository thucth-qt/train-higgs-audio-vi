#!/usr/bin/env python3
"""
Pre-startup environment fixer
Run this to fix environment issues before the main training script
"""

import os
import sys
import subprocess

def fix_torch_version_environment():
    """Fix torch version in the environment"""
    print("=" * 60)
    print("FIXING TORCH VERSION ENVIRONMENT")
    print("=" * 60)
    
    # Check if we can import torch at all
    try:
        import torch
        print(f"✓ Torch imported successfully")
        print(f"Current torch.__version__: {repr(torch.__version__)}")
        
        # Check if version is problematic
        if torch.__version__ is None or not isinstance(torch.__version__, str):
            print(f"❌ Torch version is problematic: {repr(torch.__version__)}")
            print("This will cause transformers import to fail")
            
            # Try to reinstall torch with proper version
            print("Attempting to reinstall torch...")
            cmd = "pip install --force-reinstall torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126"
            print(f"Command: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Torch reinstallation successful")
                print("Please restart Python and try again")
                return True
            else:
                print(f"❌ Torch reinstallation failed: {result.stderr}")
                return False
        else:
            print("✓ Torch version is valid")
            return True
            
    except Exception as e:
        print(f"❌ Cannot import torch: {e}")
        print("Torch installation is severely broken")
        return False

def fix_transformers():
    """Fix transformers package"""
    print("\n" + "=" * 60)
    print("FIXING TRANSFORMERS PACKAGE")
    print("=" * 60)
    
    # Try to reinstall transformers
    cmd = "pip install --force-reinstall transformers"
    print(f"Reinstalling transformers: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Transformers reinstallation successful")
        return True
    else:
        print(f"❌ Transformers reinstallation failed: {result.stderr}")
        return False

def test_fixed_environment():
    """Test if the environment is fixed"""
    print("\n" + "=" * 60)
    print("TESTING FIXED ENVIRONMENT")
    print("=" * 60)
    
    # Start a new Python process to test the fix
    test_script = '''
import torch
print(f"torch.__version__: {torch.__version__}")
print(f"Type: {type(torch.__version__)}")

if torch.__version__ is None or not isinstance(torch.__version__, str):
    print("FAIL: torch.__version__ is still invalid")
    exit(1)

try:
    import transformers
    print(f"SUCCESS: transformers imported - {transformers.__version__}")
    exit(0)
except Exception as e:
    print(f"FAIL: transformers import failed - {e}")
    exit(1)
'''
    
    result = subprocess.run([sys.executable, '-c', test_script], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Environment test passed!")
        print(result.stdout)
        return True
    else:
        print("❌ Environment test failed!")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    """Main fix routine"""
    print("Starting environment fix...")
    
    # Check if we're in virtual environment
    if 'VIRTUAL_ENV' not in os.environ and '.venv' not in sys.executable:
        print("❌ Error: Please activate your virtual environment first")
        print("Run: source .venv/bin/activate")
        return False
    
    print(f"✓ Virtual environment: {os.environ.get('VIRTUAL_ENV', sys.executable)}")
    
    # Fix torch
    torch_ok = fix_torch_version_environment()
    if not torch_ok:
        print("❌ Could not fix torch version issue")
        return False
    
    # Fix transformers
    transformers_ok = fix_transformers()
    if not transformers_ok:
        print("❌ Could not fix transformers")
        return False
    
    # Test the fixes
    test_ok = test_fixed_environment()
    if test_ok:
        print("\n🎉 ENVIRONMENT FIX SUCCESSFUL!")
        print("You can now run the training script:")
        print("./SingleGPU_training_vn_lora.sh fp16")
        return True
    else:
        print("\n❌ ENVIRONMENT FIX FAILED!")
        print("Manual intervention required")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)