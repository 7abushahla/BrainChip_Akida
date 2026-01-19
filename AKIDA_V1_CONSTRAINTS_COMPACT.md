# Akida v1 Hardware Constraints - Quick Reference

**Official Docs:** https://brainchip-inc.github.io/akida_examples_2.4.0-doc-1/user_guide/1.0_hw_constraints.html

---

## Input Dimensions

| Parameter | Constraint |
|-----------|------------|
| Width | 5 to 256 pixels |
| Height | ≥ 5 pixels |
| Channels | 1 or 3 only |

---

## Model Structure

| Requirement | Rule |
|-------------|------|
| API | Sequential or Functional (NOT subclassed Model) |
| Input Shape | Must be explicitly defined |

---

## Activations

| Aspect | Constraint |
|--------|------------|
| ReLU | Must be bounded: `ReLU(max_value=6.0)` |
| Placement | Separate layers (not fused with Conv/Dense) |
| Non-ReLU | Only in final output layer |

---

## Convolutional Layers

### InputConvolutional (First Layer)
- Kernel: 3×3, 5×5, 7×7
- Stride: 1, 2, 3
- Padding: 'same' OR 'valid'

### Convolutional (Intermediate Layers)
- Kernel: 1×1, 3×3, 5×5, 7×7
- Stride: 1 or 2 (stride=2 ONLY with 3×3)
- Padding: 'same' ONLY
- No dilated/grouped convolutions

---

## Pooling Layers

### MaxPooling
- Size: 1×1, 2×2 (InputConv also: 1×2, 2×1)
- Stride: ≤ pool size
- Padding: Must match preceding Conv layer

### GlobalAveragePooling
- Output width: ≤ 32
- Conv output height: ≥ 3 rows

---

## Layer Ordering Rules

| Rule | Description |
|------|-------------|
| Conv+MaxPool | Must be followed by another Conv (not Dense/Flatten) |
| Last Conv | Cannot have MaxPool before Dense layers |
| BatchNorm | If used, must come before activation |
| Flatten | Only directly before Dense |

---

## Quantization

### Bit Widths
| Layer Type | Input | Weights | Activation |
|------------|-------|---------|------------|
| InputConv | 8 | 8 | 1, 2, 4 |
| Conv | 1, 2, 4 | 1, 2, 4 | 1, 2, 4 |
| Dense | 1, 2, 4 | 1, 2, 4 | 1, 2, 4 |

### Data Types
- Training/Quantization: float32
- Inference: uint8

---

## Dense Layers

- Input: Must be flattened (width=1, height=1)
- Max features: ≤ 57,334

---

## Valid Block Patterns

**Convolutional:**
```
Conv → ReLU
Conv → MaxPool → ReLU
Conv → BatchNorm → ReLU
Conv → BatchNorm → ReLU → MaxPool
```

**Dense:**
```
Dense → ReLU
Flatten → Dense → ReLU
```

---

## Context Manager

Always use version context:
```python
from cnn2snn import set_akida_version, AkidaVersion

with set_akida_version(AkidaVersion.v1):
    # All operations here
```

---

## Pre-Deployment Checklist

- [ ] Input width ≤ 256
- [ ] Input height ≥ 5
- [ ] Channels = 1 or 3
- [ ] Functional/Sequential API
- [ ] All ReLU have `max_value=6.0`
- [ ] Separate activation layers
- [ ] First Conv: any valid padding
- [ ] Other Conv: `padding='same'`
- [ ] MaxPool padding matches Conv
- [ ] Last Conv has NO MaxPool
- [ ] MaxPool stride ≤ size
- [ ] Inference uses uint8
