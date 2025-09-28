import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now you can import from models folder
from models.swin_hafnet.swin_hafnet import *
import torch

if __name__ == '__main__':
    x = torch.randn(64, 3, 512, 512)  # Example input tensor
    # # Test Patch Partition and Patch Embedding
    # patch_partition = PatchPartition(patch_size=4)
    # linear_embed = LinearEmbed(in_channels=48, embed_dim=96)
    # x = patch_partition(x)
    # print("After Patch Partition:", x.shape)  # Should be (64, 48, 128, 128)
    # x = linear_embed(x)
    # print("After Linear Embedding:", x.shape)  # Should be (64, 128, 128, 96)
    # # Test Patch Merging
    # patch_merging = PatchMerging(dim=96)
    # x = patch_merging(x)
    # print("After Patch Merging:", x.shape)  # Should be (64, 64, 64, 192)
    
    # Test encoder
    encoder = Encoder()
    x = encoder(x)
    print("After Encoder:", x.shape)  # Should be (64, 384, 64, 64)