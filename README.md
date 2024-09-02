# DiT with ControlNet Support

This repository features an optimized Diffusion Transformer (DiT) designed for CIFAR10 and other pixel-space flows. Key improvements include the addition of long skip connections and a final convolutional layer.

Please note that integrating MMDiT-style ControlNet with DiT using long skip connections can lead to unstable training. To address this, the DiT architecture has been modified, particularly in the order of residual additions within the ControlNet.

Checkpoint for DiT-S/2 on `CIFAR10`, $\text{FID}_{50 \text{k}} = 3.678$
- 链接: https://pan.baidu.com/s/18bLaZ5W4GI3sjyeIMu7DFA?pwd=bk8b 提取码: bk8b

``` python3
DiT(input_size=32,
    patch_size=2,
    in_channels=3,
    out_channels=3,
    hidden_size=512,
    depth=13,
    num_heads=8,
    mlp_ratio=4,
    num_classes=0,
    use_long_skip=True,
    final_conv=True)
```
