import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from utils import total_loss, jaccard_index, pixel_accuracy, sensitivity, specificity, dice_loss, DEVICE
from models.swin_hafnet.swin_hafnet import SwinHAFNet
from dataset.dataset_prepare import prepare
# Define some training parameters
EPOCHS = 50
LEARNING_RATE = 1.5e-4
BATCH_SIZE = 16
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
LOGGING_DIR = './logs'

import warnings
warnings.filterwarnings("ignore")

def train():
    swin_ds_train, swin_ds_val, swin_ds_test, _, _, _ = prepare()
    print(swin_ds_train.__len__())
    import os
    os.makedirs(LOGGING_DIR, exist_ok=True)

    #--------------Using Transformers Trainer API for training----------------#

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


    class CustomDataCollator:
        def __call__(self, features):
            # features is a list of samples from the dataset, where each sample is a tuple of tensors (image, mask)
            # Stack the images and masks and return them as a dictionary
            images = torch.stack([f[0] for f in features])
            masks = torch.stack([f[1] for f in features])
            return {"pixel_values": images, "labels": masks}


    train_args = TrainingArguments(
        output_dir="./swin_hafnet_output",
        dataloader_pin_memory=False,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE//2,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False, # Keep unused columns
        report_to="wandb",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Ensure predictions and labels are on the same device as the metrics

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


    model = SwinHAFNet().to(device=DEVICE, dtype=torch.float32)
    swin_trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=swin_ds_train, # Use the dataset directly, not the DataLoader
        eval_dataset=swin_ds_val,   # Use the dataset directly, not the DataLoader
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(), # Pass the custom data collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    swin_trainer.train()

    eval_result = swin_trainer.evaluate(eval_dataset=swin_ds_test) # Use the dataset directly, not the DataLoader
    print(f"Test set evaluation results: {eval_result}")

    # Save the trained model
    torch.save(model.state_dict(), "swin_hafnet_model.pth")

#--------------Main----------------#
if __name__ == "__main__":
    train()