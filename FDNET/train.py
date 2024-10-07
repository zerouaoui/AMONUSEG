import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import os.path
from os import path
import pandas as pd

from dataset import SegmentationDataset
from loss import *
from utils import *

import json

from time import time

from model import FDNet

model_name = 'fdnet'
model_saving_path = None
if path.exists(model_saving_path) == False:
  os.mkdir(model_saving_path)

model_saving_path = None

model_final_path = model_saving_path + model_name + '.pth.tar'
train_data_path =  'train_data.csv'
test_data_path =  'test_data.csv'

# Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 400

k_folds = 3
LOAD_MODEL = False

performance = {}

if __name__ == "__main__":

    set_seed(42)

    transform_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(),
        ],
    )

    transform_val = A.Compose(
        [
            ToTensorV2(),
        ],
    )


    performance_per_fold = {
                            'train_loss': [], 
                            'val_loss': [], 
                            'dice_train': [], 
                            'dice_val': [], 
                            'saving_epoch': None}
    best_dice = 0.

    
    print('--------------------------------')

    # Define data loaders for training data in this fold

    train_dataset = pd.read_csv(train_data_path)
    train_dataset = SegmentationDataset(dataset=train_dataset, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, shuffle=True)

    # Define data loaders for testing data in this fold

    test_dataset = pd.read_csv(test_data_path)
    test_dataset = SegmentationDataset(dataset=test_dataset,
                                        transform=transform_val)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE, shuffle=False)


    # the architecture model
    model = FDNet(in_channels=3, out_channels=1, init_features=64).to(DEVICE)

    # loss function
    loss_fn =  DiceBCELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(torch.load(model_final_path), model)

    # training loop
    for epoch in range(NUM_EPOCHS):
        print('----------- epoch number: ' + str(epoch) + ' ---------------')

        # Training Phase
        loop = tqdm(trainloader)
        total_loss = 0
        # loop of forward and backward passes
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)

            targets = targets.unsqueeze(1).to(device=DEVICE, dtype=torch.long).permute((0, 1, 2, 3))

            # forward
            imgs = data[:, 0:3, ...]
            mean_attention_map = data[:, 3:, ...]
            predictions = model(imgs, mean_attention_map)
            loss = loss_fn(predictions.float(), targets.float())
            total_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        performance_per_fold['train_loss'].append(total_loss/len(trainloader))

        # Validation Phase
        model.eval()
        loop = tqdm(testloader)
        total_loss = 0
        # loop of forward pass for test
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)
            targets = targets.unsqueeze(1).to(device=DEVICE, dtype=torch.long)

            # forward
            with torch.no_grad():
                predictions = model(data)

            loss = loss_fn(predictions.float(), targets.float())

            total_loss += loss.item()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        performance_per_fold['val_loss'].append(total_loss / len(testloader))

        # check train dice score
        dice = check_accuracy(trainloader, model, device=DEVICE)
        performance_per_fold['dice_train'].append(dice.item())

        # check test dice score
        dice, acc, IoU, precision, recall = check_accuracy(testloader, model, device=DEVICE)
        performance_per_fold['dice_val'].append(dice.item())


        # Saving model with best DICE score on validation set
        if performance_per_fold['dice_val'][-1] > best_dice:
            performance_per_fold['saving_epoch'] = epoch
            performance_per_fold['dice_val_at_saving_epoch'] = performance_per_fold['dice_val'][-1]
            best_dice = performance_per_fold['dice_val'][-1]
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, filename=model_final_path)
        model.train()

