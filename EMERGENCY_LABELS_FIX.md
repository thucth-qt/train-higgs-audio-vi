# EMERGENCY LABELS FIX - Aggressive Multi-Layer Defense

## Critical Issue Analysis
The `labels` parameter is still reaching `HiggsAudioModel.forward()` despite our previous fixes. This indicates the HuggingFace Trainer is adding `labels` at multiple points in the pipeline.

## Root Cause Investigation
From error logs:
- Input keys: `['input_ids', 'attention_mask', 'audio_features', 'audio_feature_attention_mask', 'audio_out_ids', 'audio_out_ids_start', 'audio_out_ids_start_group_loc', 'audio_in_ids', 'audio_in_ids_start', 'label_audio_ids', 'reward']`
- No `labels` in the original inputs, but somehow `labels` is being passed to the model
- This suggests HuggingFace Trainer is automatically adding `labels` somewhere in the pipeline

## AGGRESSIVE FIX STRATEGY - 5 Layers of Defense

### Layer 1: Training Step Input Filtering
```python
def training_step(self, model, inputs, num_items_in_batch=None):
    # CRITICAL: Remove labels from inputs before any processing
    if isinstance(inputs, dict) and 'labels' in inputs:
        logger.warning("Found 'labels' in training_step inputs - removing it!")
        inputs = {k: v for k, v in inputs.items() if k != 'labels'}
```

### Layer 2: _prepare_inputs() Override
```python
def _prepare_inputs(self, inputs):
    inputs = super()._prepare_inputs(inputs)
    if isinstance(inputs, dict) and 'labels' in inputs:
        logger.debug("Removing 'labels' from inputs before model forward pass")
        inputs = {k: v for k, v in inputs.items() if k != 'labels'}
    return inputs
```

### Layer 3: compute_loss() Label Extraction
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Extract labels before preparing model inputs
    labels = None
    if isinstance(inputs, dict):
        labels = inputs.pop('labels', None)  # Remove labels from inputs dict
```

### Layer 4: _prepare_model_inputs_from_dict() Filtering
```python
def _prepare_model_inputs_from_dict(self, inputs):
    for key, value in inputs.items():
        # Skip labels entirely - they should not go to the model
        if key == 'labels':
            continue  # Don't add labels to model inputs
```

### Layer 5: Model Wrapper Forward() Final Defense
```python
def forward(self, **kwargs):
    # AGGRESSIVE DEBUG: Check for labels in kwargs
    if 'labels' in kwargs:
        logger.error(f"FOUND LABELS IN FORWARD METHOD! Input keys: {list(kwargs.keys())}")
        logger.error("This should not happen with our fixes!")
    
    # Extract labels/label_ids if present
    labels = kwargs.pop('labels', None)
    label_ids = kwargs.pop('label_ids', None)
    
    # Final safety check - ensure no labels are in kwargs
    if 'labels' in kwargs or 'label_ids' in kwargs:
        logger.error("CRITICAL: Labels still found in kwargs after removal!")
        kwargs = {k: v for k, v in kwargs.items() if k not in ['labels', 'label_ids']}
        logger.error(f"Cleaned kwargs: {list(kwargs.keys())}")
```

## Enhanced Debugging Features

### Comprehensive Logging
- `logger.error()` for critical label detection
- Input keys logging at each layer
- Device tracking for all tensors
- Model call parameter verification

### Error Detection
- Aggressive checks for `labels` presence
- Multiple removal attempts with logging
- Final safety nets before model calls

## Expected Behavior After Emergency Fix

### ✅ Defense Cascade:
```
HF Trainer → Adds 'labels' →
Layer 1 (training_step) → Removes 'labels' →
Layer 2 (_prepare_inputs) → Removes 'labels' →  
Layer 3 (compute_loss) → Extracts 'labels' →
Layer 4 (model_inputs_prep) → Skips 'labels' →
Layer 5 (forward) → Final 'labels' removal →
HiggsAudioModel.forward() ← SUCCESS: Clean inputs
```

### 🔍 Debug Output Expected:
- `"Found 'labels' in training_step inputs - removing it!"` - Shows where labels are detected
- `"FOUND LABELS IN FORWARD METHOD!"` - Should NOT appear if fixes work
- `"Calling HiggsAudioModel with: [clean_input_keys]"` - Confirms clean inputs

## Emergency Status
This is a **5-layer aggressive defense** system designed to catch and remove `labels` at every possible point in the training pipeline. Even if HuggingFace Trainer adds `labels` at multiple points, this system will detect and remove them.

**If this doesn't work**, the issue is deeper in the HuggingFace framework and may require disabling automatic label handling entirely.

The training should now succeed with comprehensive label removal! 🚀