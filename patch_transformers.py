#!/usr/bin/env python3
"""
Emergency patch for transformers import issue
This patches the version parsing to handle the None version issue
"""

def patch_torch_version():
    """Patch the torch version issue before importing transformers"""
    import torch
    
    # Check if torch.__version__ is problematic
    if torch.__version__ is None or not isinstance(torch.__version__, str):
        print(f"⚠️  Warning: torch.__version__ is invalid: {torch.__version__}")
        print("Applying emergency patch...")
        
        # Patch torch.__version__ to a valid string
        original_version = torch.__version__
        torch.__version__ = "2.7.1+cu126"  # Use the version from the log
        print(f"Patched torch.__version__ from {original_version} to {torch.__version__}")
        
        return True
    else:
        print(f"✓ torch.__version__ is valid: {torch.__version__}")
        return False

def test_imports():
    """Test imports after patching"""
    try:
        print("Testing transformers import after patch...")
        import transformers
        print(f"✓ Transformers imported successfully: {transformers.__version__}")
        
        from transformers import AutoTokenizer, AutoConfig
        print("✓ AutoTokenizer and AutoConfig imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import still failing: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("EMERGENCY TRANSFORMERS PATCH")
    print("=" * 60)
    
    # Apply patch
    patched = patch_torch_version()
    
    # Test imports
    success = test_imports()
    
    if success:
        print("\n🎉 Patch successful! You can now import transformers.")
        print("Note: This is a temporary fix. Consider updating packages for a permanent solution.")
    else:
        print("\n❌ Patch failed. Manual troubleshooting required.")
        print("Try running: python fix_transformers_issue.py")