#!/usr/bin/env python3
"""
Pre-import patch module
Import this module before any transformers imports to fix the torch version issue
"""

import torch
import sys

# Apply patch immediately when this module is imported
def _apply_immediate_patch():
    """Apply torch version patch immediately and aggressively"""
    original = torch.__version__
    fixed_version = "2.7.1+cu126"
    
    # Always patch, regardless of current value
    if original is None or not isinstance(original, str) or original == "":
        print(f"🔧 CRITICAL PATCH: Fixed invalid torch.__version__ from {repr(original)} to {repr(fixed_version)}")
        torch.__version__ = fixed_version
    else:
        # Even if it seems valid, make sure it's the expected format
        try:
            from packaging import version
            version.parse(original)  # Test if it can be parsed
            print(f"✓ torch.__version__ is valid: {repr(original)}")
        except:
            print(f"🔧 PARSE PATCH: Fixed unparseable torch.__version__ from {repr(original)} to {repr(fixed_version)}")
            torch.__version__ = fixed_version
    
    # Also patch any other version attributes that might exist
    version_attrs = ['_version', '__version_info__']
    for attr in version_attrs:
        if hasattr(torch, attr):
            old_val = getattr(torch, attr)
            if old_val is None or (isinstance(old_val, str) and old_val == ""):
                setattr(torch, attr, fixed_version)
                print(f"🔧 ATTR PATCH: Fixed torch.{attr} from {repr(old_val)} to {repr(fixed_version)}")
    
    return torch.__version__

# Apply patch when module is imported
_patched_version = _apply_immediate_patch()

print(f"🔧 TORCH PATCH MODULE: Final torch.__version__ = {repr(_patched_version)}")

# Export the patched version for verification
__all__ = ['torch', '_patched_version']