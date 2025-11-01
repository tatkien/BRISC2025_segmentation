import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_loss(logits, targets, threshold=0.5, smooth=1e-6):
    """
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes (background and tumor)
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask (tumor=1, background=0)
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size, 2, H, W)
    # Extract tumor predictions from second channel
    probs = probs[:, 1:2, ...]  # (batch_size, 1, H, W)
    preds = torch.where(probs >= threshold, 1, 0).int()
    targets = targets.int()
    # calculate Dice Score
    dice = (2 * (preds * targets) + smooth).sum() / ((preds + targets) + smooth).sum()
    return 1 - dice

def binary_cross_entropy_loss(logits, targets):
    """
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    bce_loss = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    targets = targets.float()
    return bce_loss(logits[:, 1:2, :, :].float(), targets)

def total_loss(logits, targets, smooth=1e-6, dice_weight=1.0, bce_weight=0.5):
    """
    logits: Model output tensor (batch_size, 2, H, W) - raw logits
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    dice = dice_loss(logits, targets, smooth)
    bce = binary_cross_entropy_loss(logits, targets)
    return dice_weight * dice + bce_weight * bce

def pixel_accuracy(logits, targets, threshold=0.5):
    """
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size, 2, H, W)
    preds = (probs[:, 1:2, :, :] >= threshold).int()  # (batch_size, 1, H, W)
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

def jaccard_index(logits, targets, threshold=0.5, smooth=1e-6, device=DEVICE):
    """
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['out']
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size, 2, H, W)
    preds = (probs[:, 1:2, :, :] >= threshold).int()  # (batch_size, 1, H, W)
    targets = targets.int()
    jaccard = BinaryJaccardIndex().to(device=device)(preds, targets)
    return jaccard


def sensitivity(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Also known as Recall or True Positive Rate
    Sensitivity = True Positives / (True Positives + False Negatives)
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size, 2, H, W)
    preds = (probs[:, 1:2, :, :] >= threshold).float()  # (batch_size, 1, H, W)
    targets = targets.float()

    true_positives = (preds * targets).sum()
    possible_positives = targets.sum()

    sens = (true_positives + smooth) / (possible_positives + smooth)
    return sens

def specificity(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Also known as True Negative Rate
    Specificity = True Negatives / (True Negatives + False Positives)
    Specificity = True Negatives / (True Negatives + False Positives)
    logits: Model output tensor (batch_size, 2, H, W) - raw logits for 2 classes
    targets: Ground truth tensor (batch_size, 1, H, W) for binary mask
    """
    if isinstance(logits, dict):
      logits = logits['logits']
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size, 2, H, W)
    preds = (probs[:, 1:2, :, :] >= threshold).float()  # (batch_size, 1, H, W)
    targets = targets.float()

    true_negatives = ((1 - preds) * (1 - targets)).sum()
    possible_negatives = (1 - targets).sum()

    spec = (true_negatives + smooth) / (possible_negatives + smooth)
    return spec

def evaluate(model, dataloader, device):
    """
    Evaluate model on validation/test data
    images: (batch_size, 3, H, W)
    masks: (batch_size, 1, H, W)
    model output: (batch_size, 2, H, W) - logits
    """
    model.eval()
    total_loss_value = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)  # (batch_size, 2, H, W) - raw logits
            loss = total_loss(logits, masks).item()

            total_loss_value += loss * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss_value / total_samples

    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    total_jaccard = 0.0
    total_pixel_acc = 0.0
    total_sensitivity = 0.0
    total_specificity = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)  # (batch_size, 2, H, W) - raw logits

            batch_size = images.size(0)
            total_jaccard += jaccard_index(logits, masks).item() * batch_size
            total_pixel_acc += pixel_accuracy(logits, masks).item() * batch_size
            total_sensitivity += sensitivity(logits, masks).item() * batch_size
            total_specificity += specificity(logits, masks).item() * batch_size
            total_samples += batch_size

    avg_jaccard = total_jaccard / total_samples
    avg_pixel_acc = total_pixel_acc / total_samples
    avg_sensitivity = total_sensitivity / total_samples
    avg_specificity = total_specificity / total_samples

    return {
        "jaccard": avg_jaccard,
        "pixel_accuracy": avg_pixel_acc,
        "sensitivity": avg_sensitivity,
        "specificity": avg_specificity
    }