import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
def dice_loss(probs, targets, threshold=0.5):
    """
    prob: Predicted tensor (batch_size, H, W, 2) for 2 classes (tumor and background)
    targets: Ground truth tensor (batch_size, H, W, 1) for binary mask (tumor=1, background=0)
    """
    # Convert prob to one-hot encoding
    # Note that first channel in probs is for background, second channel is for tumor
    # So we take the second channel for tumor prediction
    probs = probs[..., 1].unsqueeze(-1)  # (batch_size, H, W, 1)
    preds = torch.where(probs >= threshold, 1, 0).int()
    targets = targets.int()
    # Calculate Dice Score
    dice = DiceScore(num_classes=2, input_format='one-hot')(preds, targets)
    return 1 - dice

def binary_cross_entropy_loss(probs, targets):
    return F.binary_cross_entropy(probs[..., 1].unsqueeze(-1), targets)

def total_loss(probs, targets, smooth=1e-6):
    dice = dice_loss(probs, targets, smooth)
    bce = binary_cross_entropy_loss(probs, targets)
    return dice + bce

def pixel_accuracy(probs, targets, threshold=0.5):
    preds = (probs[..., 1].unsqueeze(-1) > threshold).float()
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

def jaccard_index(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs[..., 1].unsqueeze(-1) > threshold).int()
    targets = targets.int()
    jaccard = BinaryJaccardIndex()(preds, targets)
    return jaccard
    
    
def sensitivity(probs, targets, threshold=0.5, smooth=1e-6):
    # Also known as Recall or True Positive Rate
    # Sensitivity = True Positives / (True Positives + False Negatives)
    # True Positives: preds=1, targets=1
    # False Negatives: preds=0, targets=1
    preds = (probs[..., 1].unsqueeze(-1) > threshold).float()
    targets = targets.float()
    
    true_positives = (preds * targets).sum()
    possible_positives = targets.sum()
    
    sens = (true_positives + smooth) / (possible_positives + smooth)
    return sens

def specificity(probs, targets, threshold=0.5, smooth=1e-6):
    # Also known as True Negative Rate
    # Specificity = True Negatives / (True Negatives + False Positives)
    # True Negatives: preds=0, targets=0
    # False Positives: preds=1, targets=0
    preds = (probs[..., 1].unsqueeze(-1) > threshold).float()
    targets = targets.float()
    
    true_negatives = ((1 - preds) * (1 - targets)).sum()
    possible_negatives = (1 - targets).sum()
    
    spec = (true_negatives + smooth) / (possible_negatives + smooth)
    return spec

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = total_loss(preds, masks).item()

            total_loss += loss * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples

    return avg_loss