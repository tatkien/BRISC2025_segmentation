from models.swin_hafnet.constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import deform_conv2d
from timm.models.swin_transformer import SwinTransformerBlock
from typing import Tuple, Optional


#-------------------Encoder Blocks-------------------#  
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
    def __init__(self, in_channels=CHANNELS * PATCH_SIZE * PATCH_SIZE, embed_dim=EMBEDDING_DIM, norm_layer=nn.LayerNorm):
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
    """
    Encoder pathway
    Input shape: (B, H, W, 3)
    Output shape: (B, H/32, W/32, 384) if input_size=512 and patch_size=4
    Also outputs three intermediate features for skip connections:
    to_haf_1: (B, H/4, W/4, 96)
    to_haf_2: (B, H/8, W/8, 192)
    to_haf_3: (B, H/16, W/16, 384)

    Workflow of encoder:
    (B, H, W, 3) - Patch partition-> (B, H/4, W/4, 48) - Linear embed -> (B, H/4, W/4, 96)
        -> Swin blocks (x2) -> (B, H/4, W/4, 96) -> to_haf_1
        -> Patch Merging -> (B, H/8, W/8, 192)
        -> Swin blocks (x2) -> (B, H/8, W/8, 192) -> to_haf_2
        -> Patch Merging -> (B, H/16, W/16, 384)
        -> Swin blocks (x2) -> (B, H/16, W/16, 384) -> to_haf_3
        -> Patch Merging -> (B, H/32, W/32, 768)
        -> Swin blocks (x2) -> (B, H/32, W/32, 768)

    Finally, flatten the spatial dimensions for next stage. 
    Result shape: (B, H/32*W/32, 768)
    -> 32x downsampling if input_size=512 and patch_size=4
    """
    def __init__(self, in_channels=CHANNELS, embed_dim=EMBEDDING_DIM, depths=DEPTHS, num_heads=NUM_HEADS, patch_size=PATCH_SIZE, input_size=INPUT_SIZE):
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
                    window_size=WINDOW_SIZE,
                    shift_size=0 if (j % 2 == 0) else WINDOW_SIZE // 2,
                    mlp_ratio=MLP_RATIO,
                    qkv_bias=QKV_BIAS,
                    attn_drop=ATTN_DROP_RATE,
                    drop_path=DROP_PATH_RATE,
                    norm_layer=nn.LayerNorm
                )
                layer.append(block)
            self.layers.append(layer)
            
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
                if C == EMBEDDING_DIM:
                    self.to_haf_1 = x
                elif C == EMBEDDING_DIM * 2:
                    self.to_haf_2 = x
                elif C == EMBEDDING_DIM * 4:
                    self.to_haf_3 = x
                    
            else:
                # Process PatchMerging layer (expects (B, H, W, C))
                x = layer(x)
    
        # Flatten spatial dimensions (B, H/32*W/32, 768)
        x = x.view(x.shape[0], -1, x.shape[-1])  
        return x


#-------------------Skip Connection Blocks-------------------#
class HAF(nn.Module):
    """
    Hierarchical Attention Fusion (HAF) Block
    Input shape: x_decoder: (B, H, W, C);
                x_skip: (B, H, W, C)
    Output shape: (B, H, W, C)
    x_decoder -> swin transformer block
                    |
    x_skip ------> cat --> (B, H, W, 2C) --> conv 1x1 -> result
    """
    def __init__(self, dim, input_resolution : Tuple[int, int], num_heads=NUM_HEADS, embedding_dim=EMBEDDING_DIM):
        # dim is C in the docstring
        # resolution is (H, W)
        super(HAF, self).__init__()
        self.swin_block = None
        for i in range(2):
            self.swin_block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads[0] if dim == embedding_dim else num_heads[1] if dim == embedding_dim * SCALE_FACTOR else num_heads[2],
                window_size=WINDOW_SIZE,
                shift_size=0 if (i % 2 == 0) else WINDOW_SIZE // 2,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                attn_drop=ATTN_DROP_RATE,
                drop_path=DROP_PATH_RATE,
                norm_layer=nn.LayerNorm
            )
        self.conv1x1 = nn.Conv2d(2 * dim, dim, kernel_size=1)
    
    def forward(self, x_decoder, x_skip):
        x = self.swin_block(x_decoder)  # (B, H, W, C)
        x = torch.cat([x, x_skip], dim=-1)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)
        x = self.conv1x1(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        return x


#-------------------Decoder Blocks-------------------#

   
class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer that upsamples the input feature map.
    Input shape: (B, H, W, C)
    Output shape: (B, 2*H, 2*W, C/2)
    """
    def __init__(self, dim, dim_scale=SCALE_FACTOR):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        x = self.expand(x)  # (B, H, W, 2*C)
        x = x.view(B, H, W, 2, 2, C // 2)  # (B, H, W, 2, 2, C/2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 2, W, 2, C/2)
        x = x.view(B, H * 2, W * 2, C // 2)  # (B, 2*H, 2*W, C/2)
        x = self.norm(x)  # (B, 2*H, 2*W, C/2)
        return x
    
class ACA(nn.Module):
    """
    Adaptive Contextual Aggregator (ACA) Block
    Input shape: (B, H, W, C)
    Output shape: (B, H, W, C)
    
    Workflow:
            --> Swin Transformer Block      |
            |               
    (B, H, W, C) ->                      -->cat (B, H, W, 3C) --> conv 1 x 1 -> (B, H, W, C)    
            | --> DeformableConv2d          |
            
    """
    def __init__(self, dim, input_resolution : Tuple[int, int], depth=2, num_heads=NUM_HEADS, embedding_dim=EMBEDDING_DIM):
        super(ACA, self).__init__()
        self.swin_layers = nn.ModuleList()
        for i in range(depth):
            swin_block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads[0] if dim == embedding_dim else num_heads[1] if dim == embedding_dim * SCALE_FACTOR else num_heads[2],
                window_size=WINDOW_SIZE,
                shift_size=0 if (i % 2 == 0) else WINDOW_SIZE // 2,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                attn_drop=ATTN_DROP_RATE,
                drop_path=DROP_PATH_RATE,
                norm_layer=nn.LayerNorm
            )
            self.swin_layers.append(swin_block)
        self.conv1x1 = nn.Conv2d(3 * dim, dim, kernel_size=1)
    
    def forward(self, x_decoder, deform_group=1, deform_kernel_size=3):
        B, H , W, C = x_decoder.shape
        for layer in self.swin_layers:
            x_swin = layer(x_decoder)  # (B, H, W, C)
        x_deform = deform_conv2d(input=x_decoder.permute(0, 3, 1, 2), # (B, C, H, W)
                                 offset=torch.zeros(B, 2*deform_group*deform_kernel_size*deform_kernel_size, H, W, device=x_decoder.device),
                                 weight=torch.ones(C, C // deform_group , deform_kernel_size, deform_kernel_size, device=x_decoder.device) / (deform_kernel_size*deform_kernel_size),
                                 padding=(deform_kernel_size//2, deform_kernel_size//2),)
        x_deform = x_deform.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = torch.cat([x_decoder, x_swin, x_deform], dim=-1)  # (B, H, W, 3C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3C, H, W)
        x = self.conv1x1(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        return x


class Decoder (nn.Module):
    """
    Decoder pathway
    Input shape: (B, H/32*W/32, 768) if input_size=512 and patch_size=4
    Output shape: (B, H/2, W/2, 48) if input_size=512 and patch_size=4
    """
    def __init__(self, in_channels=SCALE_FACTOR**(len(DEPTHS))*EMBEDDING_DIM, depths=DEPTHS, input_size=INPUT_SIZE, patch_size=PATCH_SIZE, embedding_dim=EMBEDDING_DIM):
        """
        in_channels: channels of the input feature map (from encoder) 
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.to_haf_1 = None
        self.to_haf_2 = None
        self.to_haf_3 = None
        self.haf_1 = HAF(dim=embedding_dim,
                            input_resolution=(input_size // (patch_size), input_size // (patch_size)),
                            embedding_dim=embedding_dim)
        self.haf_2 = HAF(dim=embedding_dim * SCALE_FACTOR,
                            input_resolution=(input_size // (patch_size * SCALE_FACTOR), input_size // (patch_size * SCALE_FACTOR)),
                            embedding_dim=embedding_dim)
        self.haf_3 = HAF(dim=embedding_dim * SCALE_FACTOR**2,
                            input_resolution=(input_size // (patch_size * SCALE_FACTOR**2), input_size // (patch_size * SCALE_FACTOR**2)),
                            embedding_dim=embedding_dim)
        current_resolution =  input_size // (patch_size * (SCALE_FACTOR ** len(DEPTHS)))  # Starting resolution
        for depth in depths:
            self.layers.append(PatchExpanding(in_channels))
            in_channels //= 2  # Halve the channels after each upsampling
            current_resolution *= 2  # Double the resolution after each upsampling
            
            layer = nn.ModuleList()
            layer.append(ACA(in_channels, input_resolution=(current_resolution, current_resolution)))
            self.layers.append(layer)
        
        self.layers.append(PatchExpanding(in_channels))
        in_channels //= 2  # Final halving of channels
        current_resolution *= 2  # Final doubling of resolution

    def forward(self, x, x_encode_haf_1, x_encode_haf_2, x_encode_haf_3):
        # x: (B, H/32*W/32, 768) if input_size=512, patch_size=4 and DEPTHS=[2,2,2]
        B, _, C = x.shape
        x = x.view(B, int((INPUT_SIZE // (PATCH_SIZE * (SCALE_FACTOR ** len(DEPTHS))))), int((INPUT_SIZE // (PATCH_SIZE * (SCALE_FACTOR ** len(DEPTHS))))), C)  # (B, H/32, W/32, 768)
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                # Process ACA blocks
                for block in layer:
                    B, H, W, C = x.shape
                    x = block(x)  # (B, H, W, C)
                if C == EMBEDDING_DIM * (SCALE_FACTOR ** 2):
                    self.to_haf_3 = self.haf_3(x, x_encode_haf_3)  # (B, H, W, 384)
                elif C == EMBEDDING_DIM * SCALE_FACTOR:
                    self.to_haf_2 = self.haf_2(x, x_encode_haf_2)  # (B, H, W, 192)
                elif C == EMBEDDING_DIM:
                    self.to_haf_1 = self.haf_1(x, x_encode_haf_1)  # (B, H, W, 96)

            else: # PatchExpanding layer
                if C == EMBEDDING_DIM * (SCALE_FACTOR ** 2):
                    x += self.to_haf_3
                elif C == EMBEDDING_DIM * SCALE_FACTOR:
                    x += self.to_haf_2
                elif C == EMBEDDING_DIM:
                    x += self.to_haf_1
                x = layer(x)  # (B, 2*H, 2*W, C/2)
        return x  # (B, H/2, W/2, 48)
    

#-------------------Contextual Bottleneck Enhancer (CBE) Blocks-------------------#

class CBE(nn.Module):
    """
    Contextual Bottleneck Enhancer (CBE) Block
    Input shape: (B, H * W, C)
    Output shape: (B, H * W, C)
    Workflow:
    x --> Shifted along width axis followed by linear projection
        --> Shifted MLP layer (across width axis) --> depthwise conv 3x3 --> Gelu activation
        --> Shifted along width axis followed by linear projection
        --> Normalized (called x_shifted_w)
    Residual connection: x + x_shifted_w
    """
    def __init__(self, in_channels=SCALE_FACTOR**(len(DEPTHS))*EMBEDDING_DIM, token_embedding_dim=TOKEN_EMBEDDING_DIM):
        super(CBE, self).__init__()
        self.shifted_proj1 = nn.Linear(in_channels, token_embedding_dim) 
        self.shifted_mlp1 = nn.Sequential(
            nn.Linear(token_embedding_dim, token_embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_embedding_dim * 2, token_embedding_dim)
        )
        self.depthwise_conv = nn.Conv2d(token_embedding_dim, token_embedding_dim, kernel_size=3, padding=1, groups=token_embedding_dim)
        self.gelu = nn.GELU()
        self.shifted_proj2 = nn.Linear(token_embedding_dim, token_embedding_dim)
        self.shifted_mlp2 = nn.Sequential(
            nn.Linear(token_embedding_dim, token_embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_embedding_dim * 2, token_embedding_dim)
        )
        self.norm = nn.LayerNorm(token_embedding_dim)
        self.feature_proj = nn.Linear(token_embedding_dim, in_channels)
    def shift_along_width(self, x, shift_size=SHIFT_SIZE):
        """
        Shift the feature map along the width axis.
        x: (B, H, W, C)
        """
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)  # Assuming square input for simplicity
        x = x.view(B, H, W, C)
        x = torch.roll(x, shifts=shift_size, dims=2)  # Shift along width axis
        x = x.view(B, HW, C)
        return x

    def forward(self, x):
        # x: (B, H * W, C)
        x_shifted = self.shift_along_width(x)  # (B, H*W, C)
        x_shifted = self.shifted_proj1(x_shifted)  # (B, H*W, token_embedding_dim)
        x_shifted = self.shifted_mlp1(x_shifted)  # (B, H*W, token_embedding_dim)
        
        B, HW, C = x_shifted.shape
        H = W = int(HW ** 0.5)  # Assuming square input
        
        x_shifted = x_shifted.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, token_embedding_dim, H, W)
        x_shifted = self.depthwise_conv(x_shifted)  # (B, token_embedding_dim, H, W)
        x_shifted = self.gelu(x_shifted)  # (B, token_embedding_dim, H, W)
        x_shifted = x_shifted.permute(0, 2, 3, 1).contiguous().view(B, HW, C)  # (B, H*W, token_embedding_dim)
        
        x_shifted = self.shift_along_width(x_shifted, shift_size=-SHIFT_SIZE)  # (B, H*W, token_embedding_dim)
        x_shifted = self.shifted_proj2(x_shifted)  # (B, H*W, token_embedding_dim)
        x_shifted = self.shifted_mlp2(x_shifted)  # (B, H*W, token_embedding_dim)
        
        x_shifted = self.norm(x_shifted)  # (B, H*W, token_embedding_dim)
        x_shifted = self.feature_proj(x_shifted)  # (B, H*W, C)
        
        x = x + x_shifted 
        return x  # (B, H*W, C)

#-------------------Swin-HAFNet-------------------#
class SwinHAFNet(nn.Module):
    """
    Swin-HAFNet model
    Input shape: (B, H, W, 3)
    Output shape: (B, H, W, NUM_CLASSES)
    """
    def __init__(self, in_channels=CHANNELS, embed_dim=EMBEDDING_DIM, depths=DEPTHS, num_heads=NUM_HEADS, patch_size=PATCH_SIZE, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super(SwinHAFNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, 
                               embed_dim=embed_dim, 
                               depths=depths, 
                               num_heads=num_heads, 
                               patch_size=patch_size, 
                               input_size=input_size)
        self.cbe = CBE(in_channels=SCALE_FACTOR**(len(depths))*embed_dim)
        self.decoder = Decoder(in_channels=SCALE_FACTOR**(len(depths))*embed_dim, 
                               depths=depths, 
                               input_size=input_size, 
                               patch_size=patch_size, 
                               embedding_dim=embed_dim)
        self.linear_projection = nn.Linear(embed_dim // 2, num_classes)
        # Interpolation layer to upsample the output to original size
        # self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Learnable upsampling layer
        self.final_upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        x = self.encoder(x)  # (B, H/32*W/32, 768)
        x = self.cbe(x)  # (B, H/32*W/32, 768)
        x = self.decoder(x, self.encoder.to_haf_1, self.encoder.to_haf_2, self.encoder.to_haf_3)  # (B, H/2, W/2, 48)
        x = self.linear_projection(x)  # (B, H/2, W/2, num_classes)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, num_classes, H/2, W/2)
        x = self.final_upsample(x)  # (B, num_classes, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_classes)
        return x