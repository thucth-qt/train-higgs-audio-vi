#!/usr/bin/env python3
"""
Fix script for PyTorch/Transformers version compatibility
Run this in the virtual environment to fix the import issue
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{description}...")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ Failed with return code {result.returncode}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main fix routine"""
    print("=" * 60)
    print("FIXING PYTORCH/TRANSFORMERS COMPATIBILITY ISSUE")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if 'VIRTUAL_ENV' not in os.environ and '.venv' not in sys.executable:
        print("⚠️  Warning: Not running in virtual environment")
        print("Please activate your virtual environment first:")
        print("source .venv/bin/activate")
        return False
    
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
    
    # Fix sequence
    fixes = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install --upgrade packaging", "Upgrading packaging library"),
        ("pip install --force-reinstall transformers", "Reinstalling transformers"),
        ("pip install --upgrade torch torchaudio", "Upgrading PyTorch"),
    ]
    
    for cmd, description in fixes:
        success = run_command(cmd, description)
        if not success:
            print(f"❌ Fix failed: {description}")
            print("You may need to run this manually or check for other issues")
    
    # Test the fix
    print("\n" + "=" * 60)
    print("TESTING THE FIX")
    print("=" * 60)
    
    try:
        print("Testing PyTorch import...")
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        print("Testing transformers import...")
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
        
        print("Testing version parsing...")
        from packaging import version
        parsed = version.parse(torch.__version__)
        print(f"✓ Version parsing successful: {parsed}")
        
        print("\n🎉 ALL TESTS PASSED! The issue should be fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Tests still failing: {e}")
        print("\nAdditional troubleshooting needed:")
        print("1. Check for conflicting package versions")
        print("2. Consider recreating the virtual environment")
        print("3. Check system-wide package conflicts")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)