# BrainChip Akida AKD1000  
Using the AKD1000 PCIe Card installed on a Raspberry Pi Compute Module 4 Board.


## Quantization  
Provided through the QuantizeML toolkit. It performs a uniform, symmetric (zero-centered) quantization, in which a floating-point tensor $x$ is mapped to an integer tensor $x_{\mathrm{int}}$ using a single scale factor $s$ per layer for weights and activations. The quantization function $Q$ is defined as:

$$
x_{\mathrm{int}} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right),\; q_{\min},\; q_{\max}\right)
$$
