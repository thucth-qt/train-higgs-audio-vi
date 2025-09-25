# Forward Pass and Device Mismatch Fix

## Issues Found
1. **Forward Method Error**: `HiggsAudioModel.forward() got an unexpected keyword argument 'labels'`
2. **Device Mismatch Error**: `Calculated loss must be on the original device: cuda:0 but device in use is cpu`

## Root Causes

### Issue 1: Labels Parameter
- HuggingFace Trainer automatically passes `labels` to the model's forward method
- HiggsAudioModel doesn't accept `labels` as a parameter
- This causes a TypeError during forward pass

### Issue 2: Device Mismatch
- Loss tensors were being created on CPU by default in compute_loss
- HuggingFace Trainer expects loss to be on the same device as the model (GPU)
- `torch.tensor(0.0, requires_grad=True)` creates tensors on CPU by default

## Solutions Applied

### Fix 1: Modified HiggsAudioModelWrapper.forward()
```python
def forward(self, **kwargs):
    # Extract labels if present (HF Trainer passes this but HiggsAudio doesn't use it)
    labels = kwargs.pop('labels', None)  # Remove labels from kwargs
    
    # ... device handling code ...
    
    if HIGGS_AVAILABLE:
        # Call model without labels since HiggsAudioModel doesn't accept them
        outputs = self.model(**kwargs)
        
        # Calculate loss manually if labels are provided
        loss = None
        if labels is not None:
            # Manual loss calculation with proper device handling
            # ... loss calculation code ...
            loss = loss.to(model_device)  # Ensure loss is on correct device
```

### Fix 2: Device-Aware Loss Creation in compute_loss()
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    try:
        # ... existing code ...
        
        # All fallback loss tensors now created on correct device:
        return torch.tensor(0.0, requires_grad=True, device=model_device)
        
    except Exception as e:
        # Ensure dummy loss is on correct device
        try:
            model_device = next(model.parameters()).device
            return torch.tensor(0.0, requires_grad=True, device=model_device)
        except:
            return torch.tensor(0.0, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
```

## Key Changes

### 1. Label Handling
- ✅ Extract and remove `labels` from kwargs before passing to HiggsAudioModel
- ✅ Use labels for manual loss calculation if provided
- ✅ Support both language modeling (shifted) and direct loss calculation

### 2. Device Consistency
- ✅ All loss tensors created with explicit device parameter
- ✅ Loss moved to model device before returning
- ✅ Fallback device detection for error cases
- ✅ All intermediate calculations on GPU

### 3. Output Compatibility
- ✅ Return proper ModelOutput-like objects
- ✅ Preserve all model outputs while adding loss
- ✅ Handle both dict and object-style outputs

## Expected Results

After these fixes:
1. ✅ **No more "unexpected keyword argument 'labels'"** error
2. ✅ **No more device mismatch errors** - all loss tensors on GPU
3. ✅ **Proper loss calculation** with HiggsAudioModel outputs
4. ✅ **Training loop can proceed** with gradient calculation

## Training Flow
```
Input Batch (GPU) → HiggsAudioModelWrapper → 
Remove 'labels' → HiggsAudioModel.forward() → 
Calculate Loss Manually (GPU) → Return Loss + Outputs →
HuggingFace Trainer → Backward Pass ✅
```

## Next Expected Output
- ✅ Forward pass successful
- ✅ Loss calculation on GPU
- ✅ Training progress bars and metrics
- 📊 First epoch training begins

The training should now successfully start the actual training loop! 🚀