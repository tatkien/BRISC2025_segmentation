import sys
import os

# Add parent directory to Python path to access models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from models.swin_hafnet.swin_hafnet import SwinHAFNet

# Simple preprocessing: resize to 512, convert to tensor and normalize
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path=None, device="cpu", model_type="swin"):
    """
    Load either UNet or SwinHAFNet model
    
    Args:
        model_path: Path to model checkpoint
        device: 'cpu' or 'cuda'
        model_type: 'unet' or 'swin' (inferred from path if not specified)
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    
    # Infer model type from path if not explicitly specified
    if model_path and 'unet' in model_path.lower():
        model_type = 'unet'
    elif model_path and 'swin' in model_path.lower():
        model_type = 'swin'
    
    # Create the appropriate model
    if model_type == 'unet':
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,  # Don't load ImageNet weights for inference
            in_channels=3,
            classes=2,
            aux_params=None
        )
    else:  # swin
        model = SwinHAFNet()
    
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    
    # Load checkpoint if provided
    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(state, dict):
                # Try different keys
                if "model_state_dict" in state:
                    state_dict = state["model_state_dict"]
                elif "state_dict" in state:
                    state_dict = state["state_dict"]
                else:
                    state_dict = state
            else:
                state_dict = state
            
            # Remove 'unet.' or 'model.' prefixes if present
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('unet.', '').replace('model.', '')
                new_state_dict[new_key] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded checkpoint from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {model_path}: {e}")
            print("Using model with random weights")
    else:
        print(f"No checkpoint found at {model_path}, using model with random weights")
    
    return model


def preprocess_image(pil_img: Image.Image, device="cpu"):
    """
    Preprocess PIL image for model input
    
    Args:
        pil_img: PIL Image in RGB format
        device: Target device ('cpu' or 'cuda')
    
    Returns:
        Preprocessed tensor of shape (1, 3, 512, 512)
    """
    img = transform(pil_img)
    img = img.unsqueeze(0).to(device)
    return img


def predict_masks(model, input_tensor):
    """
    Run model inference and return softmax probabilities
    
    Args:
        model: Loaded model (UNet or SwinHAFNet)
        input_tensor: Preprocessed image tensor (1, 3, H, W)
    
    Returns:
        Probability tensor of shape (1, 2, H, W) after softmax
    """
    with torch.no_grad():
        logits = model(input_tensor)
        # Handle different output formats
        if isinstance(logits, dict):
            logits = logits['logits']
        probs = torch.softmax(logits, dim=1)
    return probs
