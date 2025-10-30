import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import cv2 as cv
import numpy as np
import pandas as pd
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare():
    class SWIN_HAF_dataset(Dataset):
        def __init__(self, df, threshold=245, transform=None):
            self.df = df.reset_index(drop=True)
            self.transform = transform # Augmentation defines later
            self.threshold = threshold
        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image = cv.imread(self.df.iloc[idx]['image_path'], cv.COLOR_BGR2RGB)
            mask = cv.imread(self.df.iloc[idx]['mask_path'], cv.IMREAD_GRAYSCALE)

            mask = np.where(mask>self.threshold, 1, 0)
            mask = mask.astype(np.uint8)
            
            
            image = cv.resize(image, (512,512), interpolation=cv.INTER_LINEAR)
            mask = cv.resize(mask, (512,512), interpolation=cv.INTER_NEAREST)

            if len(image.shape) ==2:
                image = np.stack([image]*3, axis=0)
            else:
                image = np.transpose(image, (2,0,1))
            image = image / 255.0
            if self.transform:
                image = self.transform(image)
        
            return torch.tensor(image, device=DEVICE), torch.tensor(mask, device=DEVICE).unsqueeze(0)

    threshold_mask = 245

    df_train = pd.read_csv(os.path.join(os.getcwd() ,'dataset/train.csv'))
    df_val = pd.read_csv(os.path.join(os.getcwd() ,'dataset/val.csv'))
    df_test = pd.read_csv(os.path.join(os.getcwd() ,'dataset/test.csv'))
    swin_ds_train = SWIN_HAF_dataset(df_train, threshold=threshold_mask)
    swin_ds_val = SWIN_HAF_dataset(df_val, threshold=threshold_mask)
    swin_ds_test = SWIN_HAF_dataset(df_test, threshold=threshold_mask)

    BATCH_SIZE=8
    NUM_WORKERS=4
    PIN_MEMORY=True
    train_loader = None
    # DataLoader(swin_ds_train, batch_size=BATCH_SIZE,
    #                             shuffle=True,
    #                             num_workers=NUM_WORKERS,
    #                             pin_memory=PIN_MEMORY)
    val_loader = None
    # DataLoader(swin_ds_val, batch_size=BATCH_SIZE,
    #                         shuffle=False,
    #                         num_workers=NUM_WORKERS,
    #                         pin_memory=PIN_MEMORY)
    test_loader = None
    # DataLoader(swin_ds_test, batch_size=BATCH_SIZE,
    #                         shuffle=False,
    #                         num_workers=NUM_WORKERS,
    #                         pin_memory=PIN_MEMORY)
    return swin_ds_train, swin_ds_val, swin_ds_test, train_loader, val_loader, test_loader