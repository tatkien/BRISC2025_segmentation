import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock

class PatchPartition(nn.Module):
    """
    Input shape: (B, H , W, C)
    Output shape: (B, H/patch_size, W/patch_size, C*patch_size*patch_size)
    E.g. 4x downsampling if patch_size=4
    """
    def __init__(self, patch_size=4):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        """
        x: (B,H,W,C)
        """
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "H and W must be divisible by patch_size."

        x = x.view(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/ps, W/ps, ps, ps, C)
        x = x.view(B, H // self.patch_size, W // self.patch_size, -1)  # (B, H/ps, W/ps, C*ps*ps)
        return x

class LinearEmbed(nn.Module):
    """
    Linear Embedding layer that flattens the patch partition to a linear embedding.
    Input shape: (B, H/patch_size, W/patch_size, C * patch_size * patch_size)
    Output shape: (B, H/patch_size, W/patch_size, embed_dim)
    """
    def __init__(self, in_channels=48, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, H, W, C * patch_size * patch_size)
        Returns:
            x: Embedded tensor of shape (B, H, W, embed_dim)
        """
        x = self.proj(x)  # (B, H, W, embed_dim)
        x = self.norm(x)  # (B, H, W, embed_dim)
        return x

# Swin Transformer Block is imported from timm, so no need to redefine it here.

class PatchMerging(nn.Module):
    """
    Patch Merging Layer that downsamples the input feature map.
    Input shape: (B, H, W, C)
    Output shape: (B, H/2, W/2, 2*C)
    """
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even."

        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C) 
        x = self.norm(x) # (B, H/2, W/2, 4*C)
        x = self.reduction(x) # (B, H/2, W/2, 2*C)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12], patch_size=4, input_size=512):
        super(Encoder, self).__init__()
        self.patch_partition = PatchPartition(patch_size=patch_size)
        self.linear_embed = LinearEmbed(in_channels * patch_size * patch_size, embed_dim)
        # There are 3 variables to hold the outputs for skip connections (will go to HAF blocks after each stage)
        self.to_haf_1 = None
        self.to_haf_2 = None
        self.to_haf_3 = None
        # Calculate resolution after patch partition
        current_resolution = input_size // patch_size  # 512 // 4 = 128
        self.layers = nn.ModuleList()
        
        for i in range(len(depths)):
            layer = nn.ModuleList()
            for j in range(depths[i]):
                block = SwinTransformerBlock(
                    dim=embed_dim * (2 ** i),
                    input_resolution=(current_resolution, current_resolution),  # Use actual resolution
                    num_heads=num_heads[i],
                    window_size=7,
                    shift_size=0 if (j % 2 == 0) else 7 // 2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    attn_drop=0.0,
                    drop_path=0.1,
                    norm_layer=nn.LayerNorm
                )
                layer.append(block)
            self.layers.append(layer)
            
            # Add patch merging only if not the last layer
            if i < len(depths) - 1:
                self.layers.append(PatchMerging(embed_dim * (2 ** i)))
                current_resolution = current_resolution // 2  # Resolution halves after patch merging
    
    def forward(self, x):
        # Assume that patch_size == 4
        x = self.patch_partition(x)  # (B, C * 4 * 4, H/4, W/4)
        x = self.linear_embed(x)  # (B, H/4, W/4, embed_dim)
    
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                # Process Swin Transformer blocks
                for block in layer:
                    B, H, W, C = x.shape
                    x = block(x)
                if C == 96:
                    self.to_haf_1 = x
                elif C == 192:
                    self.to_haf_2 = x
                elif C == 384:
                    self.to_haf_3 = x
                    
            else:
                # Process PatchMerging layer (expects (B, H, W, C))
                x = layer(x)
        
        return x