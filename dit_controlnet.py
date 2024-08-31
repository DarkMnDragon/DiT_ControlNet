from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dit_skip import PatchEmbed, TimestepEmbedder, LabelEmbedder, DiTBlock, DiT

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class DiTControlNet(nn.Module):
    """
    DiT with controlnet
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth # depth of the controlnet, smaller than the original DiT
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0: # conditional generation
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        # fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.in_blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, skip=False)
            for _ in range(depth)])
        
        self.mid_block = DiTBlock(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, skip=False)

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.in_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))

        self.controlnet_mid_block = nn.ModuleList([zero_module(nn.Linear(self.hidden_size, self.hidden_size))])
        
        self.controlnet_x_embedder = zero_module(PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True))

    @classmethod
    def from_transformer(
        cls,
        transformer: DiT,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: int = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 0,
        load_weights_from_transformer: bool = True,
    ):
        controlnet = cls(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
        )

        if load_weights_from_transformer:
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.t_embedder.load_state_dict(transformer.t_embedder.state_dict())
            if num_classes > 0:
                controlnet.y_embedder.load_state_dict(transformer.y_embedder.state_dict())
            controlnet.pos_embed.data = transformer.pos_embed.data
            missing_keys, unexpected_keys = controlnet.in_blocks.load_state_dict(transformer.in_blocks.state_dict(), strict=False)
            if len(missing_keys) > 0:
                print(f"Missing keys in controlnet.in_blocks: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys in controlnet.in_blocks: {unexpected_keys}")
            controlnet.mid_block.load_state_dict(transformer.mid_block.state_dict())
            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet
    
    def forward(
        self,
        x: torch.Tensor,
        controlnet_x: torch.Tensor,
        t: torch.Tensor,
        conditioning_scale: float = 1.0,
        y: Optional[torch.Tensor] = None,
    ):
        x = self.x_embedder(x) + self.pos_embed           # (N, T, D), where T = H * W / patch_size ** 2
        x = x + self.controlnet_x_embedder(controlnet_x)  # (N, T, D)
        t = self.t_embedder(t)                            # (N, D)

        if hasattr(self, 'y_embedder'):
            y = self.y_embedder(y, self.training)         # (N, D)
            c = t + y                                     # (N, D)
        else:
            c = t    

        in_block_samples = ()
        for block in self.in_blocks:
            x = block(x, c)                      # (N, T, D)
            in_block_samples += (x,)

        x = self.mid_block(x, c)                 # (N, T, D)
        mid_block_sample = (x,)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(in_block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)
        
        controlnet_mid_block_sample = ()
        for mid_block_sample, controlnet_mid_block in zip(mid_block_sample, self.controlnet_mid_block): # though only one mid_block_sample
            mid_block_sample = controlnet_mid_block(mid_block_sample)
            controlnet_mid_block_sample = controlnet_mid_block_sample + (mid_block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_mid_block_sample = [sample * conditioning_scale for sample in controlnet_mid_block_sample]

        return controlnet_block_samples, controlnet_mid_block_sample