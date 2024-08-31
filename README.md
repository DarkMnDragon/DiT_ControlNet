# DiT with ControlNet Support

This repository features an optimized Diffusion Transformer (DiT) designed for CIFAR10 and other pixel-space flows. Key improvements include the addition of long skip connections and a final convolutional layer.

Please note that integrating MMDiT-style ControlNet with DiT using long skip connections can lead to unstable training. To address this, the DiT architecture has been modified, particularly in the order of residual additions within the ControlNet.
