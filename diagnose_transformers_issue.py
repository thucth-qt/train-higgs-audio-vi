#!/usr/bin/env python3
"""
Diagnose and fix PyTorch/Transformers version compatibility issue
"""

import sys
import subprocess

def check_versions():
    """Check installed package versions"""
    print("=" * 60)
    print("PACKAGE VERSION DIAGNOSIS")
    print("=" * 60)
    
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version} (type: {type(torch_version)})")
        
        # Check if version is valid
        if torch_version is None or not isinstance(torch_version, str):
            print(f"❌ PyTorch version is invalid: {torch_version}")
            return False
        else:
            print(f"✓ PyTorch version is valid")
            
    except Exception as e:
        print(f"❌ Error importing PyTorch: {e}")
        return False
    
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"Transformers version: {transformers_version}")
        print(f"✓ Transformers imported successfully")
    except Exception as e:
        print(f"❌ Error importing Transformers: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        return False
    
    try:
        import packaging
        packaging_version = packaging.__version__
        print(f"Packaging version: {packaging_version}")
    except Exception as e:
        print(f"❌ Error importing packaging: {e}")
        return False
    
    return True

def test_version_parsing():
    """Test version parsing specifically"""
    print("\n" + "=" * 60)
    print("VERSION PARSING TEST")
    print("=" * 60)
    
    try:
        import torch
        torch_version = torch.__version__
        print(f"Raw PyTorch version: {repr(torch_version)}")
        
        from packaging import version
        parsed = version.parse(torch_version)
        print(f"✓ Version parsing successful: {parsed}")
        return True
        
    except Exception as e:
        print(f"❌ Version parsing failed: {e}")
        
        # Try to fix the issue
        print("\n🔧 Attempting to fix version parsing...")
        try:
            import torch
            torch_version = str(torch.__version__) if torch.__version__ is not None else "2.0.0"
            print(f"Fixed version string: {repr(torch_version)}")
            
            from packaging import version
            parsed = version.parse(torch_version)
            print(f"✓ Fixed version parsing successful: {parsed}")
            return True
            
        except Exception as fix_error:
            print(f"❌ Fix attempt failed: {fix_error}")
            return False

def check_transformers_import():
    """Test transformers import step by step"""
    print("\n" + "=" * 60)
    print("TRANSFORMERS IMPORT TEST")
    print("=" * 60)
    
    steps = [
        ("transformers", "import transformers"),
        ("dependency_versions_check", "from transformers import dependency_versions_check"),
        ("utils.versions", "from transformers.utils.versions import require_version"),
        ("chat_template_utils", "from transformers.utils import chat_template_utils"),
        ("import_utils", "from transformers.utils import import_utils"),
    ]
    
    for step_name, import_cmd in steps:
        try:
            print(f"Testing: {import_cmd}")
            exec(import_cmd)
            print(f"✓ {step_name} imported successfully")
        except Exception as e:
            print(f"❌ {step_name} import failed: {e}")
            return False
    
    return True

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)
    
    print("1. Update packaging library:")
    print("   pip install --upgrade packaging")
    
    print("\n2. Reinstall transformers:")
    print("   pip install --force-reinstall transformers")
    
    print("\n3. Check PyTorch installation:")
    print("   pip install --upgrade torch")
    
    print("\n4. Check for version conflicts:")
    print("   pip list | grep -E '(torch|transformers|packaging)'")

if __name__ == "__main__":
    print("Starting PyTorch/Transformers compatibility diagnosis...")
    
    versions_ok = check_versions()
    parsing_ok = test_version_parsing() if versions_ok else False
    import_ok = check_transformers_import() if parsing_ok else False
    
    if not (versions_ok and parsing_ok and import_ok):
        suggest_fixes()
        sys.exit(1)
    else:
        print("\n🎉 All checks passed! No compatibility issues found.")
        sys.exit(0)