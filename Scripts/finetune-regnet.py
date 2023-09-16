"""
Fine-tuning Script for Pneumonia Classification Model

This script fine-tunes a RegNet-based deep learning model for classifying pneumonia-related medical images.
It loads a pre-trained RegNet model with ImageNet weights and fine-tunes it on a custom dataset of medical images
for pneumonia classification. The fine-tuned model is saved to a specified file path.

The custom dataset should be prepared with metadata CSV file containing image file paths and labels. It should
also have a directory containing the actual image files.

The script performs the following steps:
1. Load the pre-trained RegNet model with ImageNet weights.
2. Prepare the custom dataset using 'CustomDataset' class from 'utils.py'.
3. Split the dataset into training, validation, and test sets.
4. Create data loaders for each split.
5. Modify the model's final fully connected layer to output the desired number of classes.
6. Define the loss function and optimizer.
7. Train the model using the 'TrainLoop' class from 'utils.py'.
8. Save the fine-tuned model to a specified file path.

Usage:
    Run this script to start the fine-tuning process.
"""
from torch import save
from torch.optim import Adam
from torch.nn import Linear, BCEWithLogitsLoss
from torch.utils.data import random_split, DataLoader
from torchvision.models.regnet import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights

import sys
sys.path.append("C:\College\Projects\Breathing Problem Classification")
from utils import CustomDataset, TrainLoopv2

import warnings
warnings.filterwarnings("ignore")

def finetune():
    weights = RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
    model = regnet_y_3_2gf(weights=weights)

    train_dataset = CustomDataset("Data/Processed/train_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'])
    val_dataset = CustomDataset("Data/Processed/val_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'])

    train_loader = DataLoader(train_dataset, 16, shuffle=True)
    val_loader = DataLoader(val_dataset, 16, shuffle=True)

    num_classes = 28
    in_features = model.fc.in_features
    model.fc = Linear(in_features, num_classes)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.005)

    TrainLoopv2(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, early_stopping_rounds=20, device='cuda')

    model_path = 'Models/FinetunedRegNet.pth'

    save(model.state_dict(), model_path)

if __name__ == '__main__':
    finetune()