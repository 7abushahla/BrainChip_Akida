# BrainChip Akida AKD1000  
Using the AKD1000 PCIe Card installed on a Raspberry Pi Compute Module 4 Board.


## Quantization  
BrainChip uses the **QuantizeML** toolkit for model quantization. QuantizeML applies a **uniform, symmetric (zero-centered)** quantization scheme, where a floating-point tensor $x$ is mapped to an integer tensor $x_{\mathrm{int}}$ using a scale factor $s$:



$$
x_{\mathrm{int}} = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right)
$$

with the scale chosen from the dynamic range, calculated as follows

$$
s = \frac{\max(|x|)}{2^b - 1},
$$

and dequantization approximated as

$$
x \approx x_{\mathrm{int}} \cdot s,
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

## ANN-to-SNN Conversion
BrainChip uses the **CNN2SNN** toolkit to convert a **QuantizeML-quantized CNN** into an **Akida runtime-compatible model**. In the official workflow, the deployment pipeline is:

float CNN → QuantizeML quantized model → CNN2SNN conversion → Akida runtime execution.

The public documentation does **not** describe this conversion as a classical ANN-to-SNN transformation where ReLU activations become spike counts over \(T\) timesteps or are replaced by integrate-and-fire neurons. Instead, the converted model is executed on Akida’s **event-based runtime**, where computation is sparse and integer-only.

In the current QuantizeML workflow, the converter largely dispatches to `_convert_quantizeml(model)`, which internally calls `qml_generate_model(model)` to produce an Akida model. The resulting runtime layers are still documented as convolutional and fully connected Akida layers with optional **step-wise ReLU** or other quantized activations.

As a result, the CNN is not transformed into a classical software SNN (such as those used in frameworks like SpikingJelly or SNN Toolbox). Rather, the model is lowered to an **Akida event-domain representation**. A more accurate description of the pipeline would therefore be:

**Quantized CNN → Akida-native event-driven network**

At the processor level, Akida implements **rank coding**, where information is represented by the **time and location of events**. The hardware operates in an event-driven manner: synapses store weights, neurons integrate weighted incoming events, and an output event is generated only when the accumulated input exceeds a threshold. Zero values generate no events, enabling highly sparse and energy-efficient computation.
