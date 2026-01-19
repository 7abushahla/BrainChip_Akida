# Akida v1 Hardware Constraints - Complete List

This document contains all constraints we discovered while deploying models to Akida v1 hardware.

**Official Documentation:**
- https://brainchip-inc.github.io/akida_examples_2.4.0-doc-1/user_guide/1.0_hw_constraints.html
- https://doc.brainchipinc.com/user_guide/hardware/1.0.html

---

## 1. Input Dimensions

| Constraint | Rule | Impact |
|------------|------|--------|
| **Width** | Must be between **5 and 256 pixels** | ❌ Exceeding causes Conv layers to run in SOFTWARE (no CNP acceleration) |
| **Height** | Must be **≥ 5 pixels** | ❌ Violating prevents model conversion |
| **Channels** | Must be **1 or 3** | ❌ Other values not supported |

**Example Error:**
```
Input width 440 exceeds 256 limit → Conv layers run in software
```

---

## 2. Model API Requirements

| Constraint | Rule | Impact |
|------------|------|--------|
| **Model Type** | Must use **Sequential or Functional API** | ❌ Subclassed Model causes quantization error |
| **Input Shape** | Must be explicitly defined | ❌ Dynamic shapes not supported |

**Example Error:**
```
AttributeError: The layer "simple_cnn2" has never been called and thus has no defined input shape.
```

**Fix:**
```python
# ❌ BAD - Subclassed Model
class MyModel(keras.Model):
    def call(self, x):
        ...

# ✅ GOOD - Functional API
inputs = keras.Input(shape=(height, width, channels))
x = keras.layers.Conv2D(...)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
```

---

## 3. Activation Functions

| Constraint | Rule | Impact |
|------------|------|--------|
| **ReLU Type** | Must use **bounded ReLU** with `max_value=6.0` | ❌ Unbounded ReLU causes conversion error |
| **Activation Placement** | Must be **separate layers** (not fused) | ❌ Fused activations may cause issues |
| **Non-ReLU Activations** | Only allowed in **final layer** | ❌ Softmax/sigmoid in intermediate layers not supported |

**Example Error:**
```
ValueError: unbounded QuantizedReLU is not supported in AkidaVersion.v1
```

**Fix:**
```python
# ❌ BAD - Unbounded, fused activation
keras.layers.Conv2D(16, activation='relu')

# ✅ GOOD - Bounded, separate activation
keras.layers.Conv2D(16)
keras.layers.ReLU(max_value=6.0)
```

---

## 4. Convolutional Layer Constraints

### 4.1 InputConvolutional (First Layer)

| Parameter | Allowed Values |
|-----------|----------------|
| Kernel Size | **3×3, 5×5, 7×7** |
| Stride | **1, 2, 3** |
| Padding | **'same' OR 'valid'** |

### 4.2 Convolutional (Intermediate Layers)

| Parameter | Allowed Values |
|-----------|----------------|
| Kernel Size | **1×1, 3×3, 5×5, 7×7** |
| Stride | **1 or 2** (stride=2 ONLY with 3×3 kernels) |
| Padding | **'same' ONLY** (NOT 'valid') |

**Example Error:**
```
RuntimeError: Only padding same is supported for layer 'conv2'
```

**Fix:**
```python
# Conv block 1 (InputConv) - can use 'valid' or 'same'
x = keras.layers.Conv2D(16, kernel_size=5, padding='valid')(inputs)

# Conv block 2+ (Convolutional) - MUST use 'same'
x = keras.layers.Conv2D(16, kernel_size=5, padding='same')(x)
```

### 4.3 Not Supported
- ❌ Dilated (atrous) convolutions
- ❌ Grouped convolutions

---

## 5. Pooling Layer Constraints

### 5.1 MaxPooling

| Constraint | Rule | Impact |
|------------|------|--------|
| **Pool Size** | **1×1, 2×2** (InputConv also allows 1×2, 2×1) | ❌ Other sizes not supported |
| **Stride** | Must be **≤ pool size** | ❌ Exceeding causes error |
| **Padding** | Must **match preceding Conv layer** | ❌ Mismatch causes conversion error |

**Example Error:**
```
ValueError: Pooling layer maxpool2 (padding: valid) must have the same padding as conv2 (padding: same)
```

**Fix:**
```python
# ❌ BAD - Padding mismatch
x = keras.layers.Conv2D(16, padding='same')(x)
x = keras.layers.MaxPool2D(pool_size=2)  # Default padding='valid'

# ✅ GOOD - Matching padding
x = keras.layers.Conv2D(16, padding='same')(x)
x = keras.layers.MaxPool2D(pool_size=2, padding='same')(x)
```

### 5.2 GlobalAveragePooling

| Constraint | Rule | Impact |
|------------|------|--------|
| **Output Width** | Must be **≤ 32** | ❌ Exceeding not supported |
| **Output Height** | Must have **≥ 3 rows** | ❌ Less than 3 rows not supported |

---

## 6. Layer Ordering Constraints

### 6.1 Critical Ordering Rules

| Rule | Description | Impact |
|------|-------------|--------|
| **Conv+MaxPool Placement** | Conv with MaxPool **must be followed by another Conv** | ❌ Conv+MaxPool→Dense causes mapping error |
| **Last Conv Before Dense** | Last conv layer **cannot have MaxPool** | ❌ Violating prevents hardware mapping |
| **BatchNorm Placement** | If used, must come **before activation** | ❌ Wrong order may cause issues |
| **Flatten Placement** | Only allowed **directly before Dense** | ❌ Other placements not supported |

**Example Error:**
```
RuntimeError: A convolutional or separable convolutional layer with max pooling must be 
followed by another convolutional or separable convolutional layer. Layer 'conv4' has max pooling.
```

**Fix:**
```python
# ❌ BAD - MaxPool on last conv before Dense
Conv → MaxPool
Conv → MaxPool
Conv → MaxPool
Conv → MaxPool  ← Problem!
Flatten → Dense

# ✅ GOOD - No MaxPool on last conv
Conv → MaxPool
Conv → MaxPool
Conv → MaxPool
Conv (no pool)  ← OK!
Flatten → Dense
```

### 6.2 Valid Block Patterns

Based on error messages, these patterns are supported:

**Conv Blocks:**
```
- Conv → ReLU
- Conv → MaxPool → ReLU
- Conv → ReLU → GlobalAveragePooling
- Conv → GlobalAveragePooling
- Conv → BatchNorm → ReLU
- Conv → BatchNorm → ReLU → MaxPool
```

**Dense Blocks:**
```
- Dense → ReLU
- Flatten → Dense → ReLU
```

**Invalid Pattern:**
```
❌ Conv → MaxPool → ReLU → GlobalAveragePooling
```

---

## 7. Quantization Constraints

### 7.1 Bit Widths

| Layer Type | Input Bits | Weight Bits | Activation Bits |
|------------|------------|-------------|-----------------|
| InputConvolutional | **8** | **8** | **1, 2, 4** |
| Convolutional | **1, 2, 4** | **1, 2, 4** | **1, 2, 4** |
| Dense | **1, 2, 4** | **1, 2, 4** | **1, 2, 4** |

**Note:** CNN2SNN quantization typically uses **≥2 bits** for weights/activations because float weights are signed.

### 7.2 Input Data Type

| Constraint | Rule | Impact |
|------------|------|--------|
| **Inference Input** | Must be **uint8** | ❌ float32 causes runtime error |
| **Training/Quantization** | Can use **float32** | ✅ Converts during quantization |

**Example Error:**
```
ValueError: Input dtype should be uint8
```

**Fix:**
```python
# During training/quantization: use float32
samples_float = data.astype(np.float32) / 255.0

# For Akida inference: convert to uint8
samples_uint8 = (data * 255).astype(np.uint8)
model_akida.forward(samples_uint8)  # ✅
```

---

## 8. Dense (Fully Connected) Layer Constraints

| Constraint | Rule | Impact |
|------------|------|--------|
| **Input Dimensions** | Width = 1, Height = 1 (flattened) | ❌ Spatial dims must be collapsed |
| **Total Input Features** | Must be **≤ 57,334** | ❌ Exceeding may cause issues |

---

## 9. Version-Specific Settings

### Using Akida v1

Always wrap your code in the version context:

```python
from cnn2snn import set_akida_version, AkidaVersion

with set_akida_version(AkidaVersion.v1):
    # Create model
    model = create_model()
    
    # Quantize
    model_quantized = quantize(model, qparams=qparams, samples=samples)
    
    # Convert
    model_akida = convert(model_quantized)
    
    # Check compatibility
    check_model_compatibility(model, device=device)
```

---

## 10. Common Error Patterns

### Error 1: Layer Ordering
```
RuntimeError: A convolutional or separable convolutional layer with max pooling 
cannot be the last layer of a sequence.
```
**Fix:** Remove MaxPool from last conv block before Dense layers

### Error 2: Padding Mismatch
```
ValueError: Pooling layer maxpool2 (padding: valid) must have the same padding as conv2 (padding: same)
```
**Fix:** Add matching `padding='same'` to MaxPool2D

### Error 3: Unbounded ReLU
```
ValueError: unbounded QuantizedReLU is not supported
```
**Fix:** Use `ReLU(max_value=6.0)` instead of `activation='relu'`

### Error 4: Invalid Pattern
```
RuntimeError: Invalid block found during conversion
```
**Fix:** Check the "Compatible patterns" list in error message

### Error 5: Input Shape
```
AttributeError: The layer has never been called and thus has no defined input shape
```
**Fix:** Use Sequential or Functional API, not subclassed Model

---

## 11. Full Working Example

```python
import tensorflow as tf
from tensorflow import keras
from cnn2snn import set_akida_version, AkidaVersion, convert, check_model_compatibility
from quantizeml.models import quantize, QuantizationParams
import akida

# Create Akida v1 compatible model
def create_akida_compatible_model(num_classes=12, input_shape=(239, 220, 1)):
    """
    All constraints applied:
    - Input width ≤ 256
    - Functional API
    - Bounded ReLU
    - Correct padding
    - No MaxPool on last conv
    """
    inputs = keras.Input(shape=input_shape)
    
    # Conv1: InputConv - can use 'valid' or 'same'
    x = keras.layers.Conv2D(16, kernel_size=5, padding='valid')(inputs)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    
    # Conv2: Must use 'same'
    x = keras.layers.Conv2D(16, kernel_size=5, padding='same')(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    
    # Conv3: Must use 'same'
    x = keras.layers.Conv2D(16, kernel_size=5, padding='same')(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    
    # Conv4: NO MaxPool (last conv before Dense)
    x = keras.layers.Conv2D(32, kernel_size=5, padding='same')(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    
    # Dense layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    outputs = keras.layers.Dense(num_classes)(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Deploy to Akida
with set_akida_version(AkidaVersion.v1):
    # Create model
    model = create_akida_compatible_model()
    
    # Quantize (use float32 samples)
    qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
    model_quantized = quantize(model, qparams=qparams, samples=samples_float, num_samples=10)
    
    # Convert to Akida
    model_akida = convert(model_quantized)
    
    # Map to hardware
    devices = akida.devices()
    device = devices[0]
    model_akida.map(device)
    
    # Run inference (use uint8 samples)
    predictions = model_akida.forward(samples_uint8)
```

---

## 12. Quick Checklist

Before deploying to Akida v1, verify:

- [ ] Input width ≤ 256 pixels
- [ ] Input height ≥ 5 pixels
- [ ] Input channels = 1 or 3
- [ ] Using Sequential or Functional API (not subclassed)
- [ ] All ReLU layers have `max_value=6.0`
- [ ] Activations are separate layers (not fused)
- [ ] First Conv can use 'valid' or 'same' padding
- [ ] All other Conv layers use `padding='same'`
- [ ] MaxPool padding matches Conv padding
- [ ] Last Conv before Dense has NO MaxPool
- [ ] No dilated or grouped convolutions
- [ ] MaxPool stride ≤ pool size
- [ ] Inference uses uint8 input data
- [ ] Wrapped in `set_akida_version(AkidaVersion.v1)` context

---

## Summary

The most commonly encountered constraints (in order of discovery):

1. **Model API:** Must use Functional/Sequential, not subclassed
2. **Bounded ReLU:** Use `ReLU(max_value=6.0)` everywhere
3. **Padding:** Intermediate Conv must use `padding='same'`
4. **Padding Match:** Conv and MaxPool must have matching padding
5. **No MaxPool on Last Conv:** Remove MaxPool from conv before Dense
6. **Input Width:** Must be ≤256 for CNP hardware acceleration
7. **uint8 Input:** Inference requires uint8 data type

Following these rules ensures your model will be compatible with Akida v1 hardware!
