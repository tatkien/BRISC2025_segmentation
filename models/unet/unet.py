from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.data.data_collator import default_data_collator
from collections.abc import Mapping
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import wandb
import os
import warnings
from transformers import PretrainedConfig, PreTrainedModel

from utils import total_loss, jaccard_index, pixel_accuracy, sensitivity, specificity, dice_loss, plot_test
from dataset.dataset_prepare import prepare

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
LEARNING_RATE = 1.5e-4
BATCH_SIZE = 16
WARMUP_STEPS = 500  # 10% of num_steps
WEIGHT_DECAY = 0.01
LOGGING_DIR_UNET = './logs_unet'
LOGGING_DIR_SWIN = './logs_swin'
LOGGING_STEPS = 100

os.makedirs(LOGGING_DIR_UNET, exist_ok=True)
os.makedirs(LOGGING_DIR_SWIN, exist_ok=True)
warnings.filterwarnings("ignore")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=BATCH_SIZE):
        # inputs here will be a list of tensors from the custom data collator
        # We need to stack them to form a batch
        # Get the device from the model's parameters
        device = next(model.parameters()).device
        images = inputs['pixel_values'].to(device).float()  # Explicitly cast to float
        masks = inputs['labels'].to(device).float()  # Explicitly cast to float

        outputs = model(images)
        loss = total_loss(outputs, masks)
        return (loss, outputs) if return_outputs else loss

    # Add prediction_step method to handle the input format
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = next(model.parameters()).device
        images = inputs['pixel_values'].to(device).float()
        masks = inputs['labels'].to(device).float()

        with torch.no_grad():
            outputs = model(images)
            loss = total_loss(outputs, masks)

        # Return loss, logits, and labels
        return (loss, outputs, masks)

    # Override on_train_begin to prevent accessing model.config.to_json_string()
    def on_train_begin(self, args, state, control):

        return control


class CustomDataCollator:
    def __call__(self, features):
        # features is a list of samples from the dataset, where each sample is a tuple of tensors (image, mask)
        # Stack the images and masks and return them as a dictionary
        images = torch.stack([f[0] for f in features])
        masks = torch.stack([f[1] for f in features])
        return {"pixel_values": images, "labels": masks}


train_args_unet = TrainingArguments(
    output_dir="./unet_output",
    dataloader_pin_memory=False,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOGGING_DIR_UNET,
    logging_steps=LOGGING_STEPS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    report_to="wandb",  # Disable all logging integrations including TensorBoard
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Ensure predictions and labels are on the same device as the metrics
    if isinstance(logits, dict):
        logits = logits['logits']
    logits = torch.tensor(logits).to(DEVICE)
    labels = torch.tensor(labels).to(DEVICE)

    # Calculate metrics using the provided functions
    jaccard = jaccard_index(logits, labels).item()
    pixel_acc = pixel_accuracy(logits, labels).item()
    sens = sensitivity(logits, labels).item()
    spec = specificity(logits, labels).item()
    dice = 1 - dice_loss(logits, labels).item()


    return {
        "jaccard": jaccard,
        "pixel_accuracy": pixel_acc,
        "sensitivity": sens,
        "specificity": spec,
        "dice_score": dice
    }


# Create a wrapper class that inherits from PreTrainedModel
class UNetWrapper(PreTrainedModel):
    def __init__(self, encoder_name="efficientnet-b0", in_channels=3, classes=2):
        # Initialize with a config
        config = UNetConfig(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes
        )
        super().__init__(config)

        # Create the actual U-Net model
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            # encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            aux_params=None  # Ensure no auxiliary outputs
        )

    def forward(self, pixel_values, labels=None):
        outputs = self.unet(pixel_values)

        if labels is not None:
            # Calculate loss if labels are provided
            loss = total_loss(outputs, labels)
            return {"loss": loss, "logits": outputs}

        return {"logits": outputs}

# Config class
class UNetConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(self, encoder_name="efficientnet-b0", in_channels=3, classes=2, **kwargs):
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        super().__init__(**kwargs)

def train_unet():
    """Train UNet model"""
    # Prepare datasets
    swin_ds_train, swin_ds_val, swin_ds_test, _, _, _ = prepare()
    
    # Use the wrapped model
    unet_model = UNetWrapper(
        encoder_name="efficientnet-b0",
        in_channels=3,
        classes=2,
    ).to(device=DEVICE, dtype=torch.float32)

    unet_trainer = CustomTrainer(
        model=unet_model,
        args=train_args_unet,
        train_dataset=swin_ds_train,  # Use the dataset directly, not the DataLoader
        eval_dataset=swin_ds_val,     # Use the dataset directly, not the DataLoader
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(),  # Pass the custom data collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    unet_trainer.train()

    eval_result = unet_trainer.evaluate(eval_dataset=swin_ds_test)  # Use the dataset directly, not the DataLoader
    print(f"Test set evaluation results: {eval_result}")

    # Get df_test for plotting
    import pandas as pd
    df_test = pd.read_csv(os.path.join(os.getcwd(), 'dataset/test.csv'))
    plot_test(unet_trainer, df_test)
    
    wandb.finish()
    
    # Save the trained model
    torch.save(unet_model.state_dict(), "checkpoints/unet_model.pth")
    print("Model saved to checkpoints/unet_model.pth")


if __name__ == "__main__":
    train_unet()