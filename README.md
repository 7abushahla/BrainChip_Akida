# BrainChip Akida AKD1000  
Using the AKD1000 PCIe Card installed on a Raspberry Pi Compute Module 4 Board.

## Table of Contents

- [Akida v1 Overview](#akida-v1-overview)
- [Akida v1 Deployment Constraints (High‑Level)](#akida-v1-deployment-constraints-highlevel)
  - [Input and Model Structure](#input-and-model-structure)
  - [Convolution, Pooling, and Block Patterns](#convolution-pooling-and-block-patterns)
  - [Activations](#activations)
  - [Dense Layers](#dense-layers)
- [Quantization](#quantization)
  - [Bit‑Widths and Data Types](#bitwidths-and-data-types)
- [ANN-to-SNN Conversion](#ann-to-snn-conversion)
- [How Inference Runs on Akida Hardware](#how-inference-runs-on-akida-hardware)

## Akida v1 Overview

Akida v1 is a neuromorphic CNN accelerator that executes **event‑driven, integer‑only** networks derived from standard CNNs. Models are trained in floating point, quantized with **QuantizeML**, converted with **CNN2SNN**, and finally deployed to the Akida runtime, which runs them as sparse, event‑based networks on chip.

At the processor level, Akida implements **rank coding**, where information is represented by the **time and location of events**. Synapses store weights, neurons integrate weighted incoming events, and an output event is generated only when the accumulated input exceeds a threshold. Zero values generate no events, enabling highly sparse and energy‑efficient computation.

## Akida v1 Deployment Constraints (High‑Level)

These are the most important constraints you must satisfy *before* quantization and conversion; violating them typically results in conversion errors or forces layers to run in software only.

### Input and Model Structure

- **Input dimensions**
  - **Width**: 5–256 pixels  
  - **Height**: ≥ 5 pixels  
  - **Channels**: 1 or 3 only
- **Model API**
  - **Use**: Keras **Sequential** or **Functional** API  
  - **Avoid**: Subclassed `keras.Model` (no dynamic / undefined input shapes)
  - **Input shape** must be explicitly defined in the `Input` layer.

### Convolution, Pooling, and Block Patterns

- **First convolution (InputConvolutional)**
  - Kernel: 3×3, 5×5, or 7×7  
  - Stride: 1, 2, or 3  
  - Padding: `'same'` or `'valid'`

- **Subsequent convolutions**
  - Kernel: 1×1, 3×3, 5×5, or 7×7  
  - Stride: 1 or 2 (stride 2 only allowed with 3×3 kernels)  
  - Padding: **`'same'` only**  
  - No dilated or grouped convolutions.

- **MaxPooling**
  - Pool sizes: 1×1 or 2×2 (for the first conv, also 1×2 or 2×1)  
  - Stride: ≤ pool size  
  - Padding: **must match** the preceding Conv layer.

- **GlobalAveragePooling**
  - Output width ≤ 32  
  - Convolutional output height ≥ 3 rows.

- **Valid block patterns (examples)**
  - `Conv → ReLU`  
  - `Conv → MaxPool → ReLU`  
  - `Conv → BatchNorm → ReLU`  
  - `Conv → BatchNorm → ReLU → MaxPool`  
  - `Conv (…optional blocks…) → GlobalAveragePooling`  
  - `Flatten → Dense → ReLU`

- **Layer ordering rules**
  - A `Conv + MaxPool` block **must be followed by another Conv**, not directly by Dense/Flatten.  
  - The **last Conv before Dense must not have MaxPool**.  
  - If used, `BatchNorm` must come **before** activation.  
  - `Flatten` is only supported **directly before** Dense layers.

### Activations

- **ReLU**
  - Must be **bounded**: use `ReLU(max_value=6.0)`  
  - Should be **separate layers** (not passed as `activation='relu'` inside Conv/Dense).
- **Non‑ReLU activations**
  - Only allowed in the **final output layer** (e.g., Softmax / sigmoid).

### Dense Layers

- Input to Dense must be **flattened** (spatial width = height = 1).  
- Total input features to a Dense layer must be **≤ 57,334**.

---

## Quantization  

BrainChip uses the **QuantizeML** toolkit for model quantization. QuantizeML applies a **uniform, symmetric (zero-centered)** quantization scheme, where a floating-point tensor $x$ is mapped to an integer tensor $x_{\mathrm{int}}$ using a scale factor $s$:

$$
x_{\mathrm{int}} = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right)
$$

with the scale chosen from the dynamic range, calculated as:

$$
s = \frac{\max(|x|)}{2^b - 1},
$$

and dequantization approximated as

$$
x \approx x_{\mathrm{int}} \cdot s.
$$

In addition to this quantization mapping, QuantizeML represents layer inputs, outputs, and weights using FixedPoint (QFloat) values, where

$$
x_{\mathrm{float}} \approx x_{\mathrm{int}} \cdot 2^{-x_{\mathrm{frac\_bits}}}
$$

Here, the stored integer $x_{\mathrm{int}}$ is interpreted with a fixed number of fractional bits $x_{\mathrm{frac\_bits}}$. Increasing the number of fractional bits improves precision. This FixedPoint representation allows quantized layers to be implemented as **integer-only operations**.

Example from the QuantizeML documentation for representing $\pi$ in 8-bit FixedPoint:

| frac\_bits | $x_{\mathrm{int}}$ | represented value |
|---|---:|---:|
| 1 | 6   | 3.0 |
| 3 | 25  | 3.125 |
| 6 | 201 | 3.140625 |

### Bit‑Widths and Data Types

On Akida v1, quantized layers must respect specific bit‑widths and data types:

| Layer Type | Input bits | Weight bits | Activation bits |
|-----------|-----------:|------------:|----------------:|
| InputConv | 8          | 8           | 1, 2, 4         |
| Conv      | 1, 2, 4    | 1, 2, 4     | 1, 2, 4         |
| Dense     | 1, 2, 4    | 1, 2, 4     | 1, 2, 4         |

- **Training / Quantization**: tensors are typically `float32`.  
- **Inference on Akida**: inputs must be `uint8` (e.g., images scaled to [0, 255] and cast to `np.uint8`).

---

## ANN-to-SNN Conversion
BrainChip uses the **CNN2SNN** toolkit to convert a **QuantizeML-quantized CNN** into an **Akida runtime-compatible model**. In the official workflow, the deployment pipeline is:

float CNN → QuantizeML quantized model → CNN2SNN conversion → Akida runtime execution.

The public documentation does **not** describe this conversion as a classical ANN-to-SNN transformation where ReLU activations become spike counts over \(T\) timesteps or are replaced by integrate-and-fire neurons. Instead, the converted model is executed on Akida’s **event-based runtime**, where computation is sparse and integer-only.

In the current QuantizeML workflow, the converter largely dispatches to `_convert_quantizeml(model)`, which internally calls `qml_generate_model(model)` to produce an Akida model. The resulting runtime layers are still documented as convolutional and fully connected Akida layers with optional **step-wise ReLU** or other quantized activations.

As a result, the CNN is not transformed into a classical software SNN (such as those used in frameworks like SpikingJelly or SNN Toolbox). Rather, the model is lowered to an **Akida event-domain representation**. A more accurate description of the pipeline would therefore be:

**Quantized CNN → Akida-native event-driven / SNN-equivalent network**

At the processor level, Akida implements **rank coding**, where information is represented by the **time and location of events**. The hardware operates in an event-driven manner: synapses store weights, neurons integrate weighted incoming events, and an output event is generated only when the accumulated input exceeds a threshold. Zero or negative summed inputs generate no output event, enabling sparse and energy-efficient computation.

## How Inference Runs on Akida Hardware
For **Akida 1.0**, the public MetaTF/CNN2SNN workflow is best understood as realizing an **SNN-equivalent event-domain model**, rather than exposing a classical multi-timestep software SNN with explicit IF/LIF layers and a user-defined simulation horizon \(T\).

At the software boundary, the chip receives a **quantized tensor**, and the converted Akida model is represented using integer runtime layers. In the public API, these converted layers are still described as convolutional / fully connected Akida layers with optional **step-wise ReLU** or related quantized activations. This suggests that the converted model is not exposed as a standard timestep-unrolled SNN graph in the way common SNN frameworks do.

Several recent papers interpret this behavior as follows: Akida uses an **equivalent representation of an SNN** based on **step-wise quantized ReLU**, allowing inference to be carried out in **a single time step / single hardware pass**. In particular, one paper states that the chip “**squashes the rate-code approximation of the ReLU into one time step**,” where it is then represented by a step-wise quantized ReLU.

At the same time, the exact neuron model used for converted MetaTF/CNN2SNN models is **not clearly documented** in the currently available public documentation. A recent analysis notes that the implementation appears to reduce converted spiking computation to a **single time step**, while also arguing that the processor behavior itself is still consistent with **ROC decoding** and **integrate-and-fire-style** event processing.

A useful mental model is therefore:

**quantized sparse tensor → nonzero activations represented as events → events routed across NPUs → weighted accumulation in neurons → threshold crossing emits new events → only emitted events propagate further**

So, for converted CNNs on Akida 1.0, inference is best described as **single-step SNN-equivalent execution on event-driven neuromorphic hardware**, rather than as an explicitly exposed multi-timestep rate-coded IF/LIF simulation.
