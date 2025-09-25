# CRITICAL DEBUG: Model Signature Inspection and Parameter Filtering

## Mystery Analysis
The logs show a confusing situation:
- **Input kwargs keys**: `['input_ids', 'attention_mask', 'audio_features', 'audio_feature_attention_mask', 'audio_out_ids', 'audio_out_ids_start', 'audio_out_ids_start_group_loc', 'audio_in_ids', 'audio_in_ids_start', 'label_audio_ids', 'reward']`
- **No 'labels' in the list**, yet the error is: `HiggsAudioModel.forward() got an unexpected keyword argument 'labels'`
- **Our code shows**: `Labels device: cuda:0` - meaning our wrapper found and extracted labels

## Possible Root Causes

### 1. Hidden Label Addition
Something in the call chain is adding `labels` between our wrapper and the actual HiggsAudioModel.forward() call.

### 2. Model Wrapper Chain
There might be multiple model wrappers, and one of them is adding `labels` parameter.

### 3. HuggingFace Framework Magic
HuggingFace might have some automatic parameter injection happening at a lower level.

## AGGRESSIVE DEBUG STRATEGY

### Phase 1: Model Signature Inspection
```python
import inspect
sig = inspect.signature(self.model.forward)
logger.error(f"Model forward signature: {sig}")
```
This will show us exactly what parameters the HiggsAudioModel.forward() method accepts.

### Phase 2: Parameter Filtering
```python
# Get valid parameters from model signature
valid_params = set(sig.parameters.keys())
valid_params.discard('self')  # Remove 'self' parameter

# Filter kwargs to only include parameters that the model accepts
filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
logger.error(f"Filtered kwargs for model: {list(filtered_kwargs.keys())}")
logger.error(f"Removed parameters: {set(kwargs.keys()) - set(filtered_kwargs.keys())}")
```

### Phase 3: Direct Forward Call
```python
# Call model.forward directly with filtered parameters
outputs = self.model.forward(**filtered_kwargs)
```

## Expected Debug Output

### ✅ If this works, we should see:
```
ERROR:__main__:Model forward signature: (self, input_ids, attention_mask, audio_features, ...)
ERROR:__main__:Filtered kwargs for model: ['input_ids', 'attention_mask', 'audio_features', ...]
ERROR:__main__:Removed parameters: set() or {'labels'}
ERROR:__main__:SUCCESS: Model call completed without error
```

### 🔍 If we still get errors:
- We'll see the exact model signature
- We'll see what parameters are being removed
- We'll know if the issue is deeper in the model architecture

## Hypothesis Testing

### Hypothesis 1: Parameter Injection
If we see `Removed parameters: {'labels'}`, it means something is still adding labels to our kwargs after all our filtering.

### Hypothesis 2: Model Architecture Issue
If we still get the error even after parameter filtering, the issue might be in how the HiggsAudioModel itself is structured.

### Hypothesis 3: Wrapper Chain
If the model signature shows unexpected parameters, there might be multiple wrappers in the chain.

## Fallback Strategy

If parameter filtering works:
1. **Implement permanent parameter filtering** based on model signature
2. **Remove all ad-hoc label removal code** and use signature-based filtering
3. **Add automatic parameter validation** for all model calls

If parameter filtering doesn't work:
1. **Inspect the actual model object** (`type(self.model)`, `self.model.__class__`)
2. **Check for model wrappers** or decorators
3. **Consider bypassing HuggingFace Trainer** entirely for this model

## Critical Debug Information Needed

The next run should provide:
1. **Exact model forward signature**
2. **What parameters are being filtered out**
3. **Whether direct forward call succeeds**
4. **Any remaining error messages**

This will finally reveal where the `labels` parameter is coming from! 🔍