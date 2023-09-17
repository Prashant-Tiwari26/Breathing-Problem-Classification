"""
Fine-tuning Script for Pneumonia Classification Model

This script fine-tunes a EfficientNetv2-based deep learning model for classifying pneumonia-related medical images.
It loads a pre-trained EfficientNetv2 model with ImageNet weights and fine-tunes it on a custom dataset of medical images
for pneumonia classification. The fine-tuned model is saved to a specified file path.

The custom dataset should be prepared with metadata CSV file containing image file paths and labels. It should
also have a directory containing the actual image files.

The script performs the following steps:
1. Load the pre-trained EfficientNetv2 model with ImageNet weights.
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
from torch.nn import Linear, BCEWithLogitsLoss, Sequential, ReLU, Dropout
from torch.utils.data import random_split, DataLoader
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights

import sys
sys.path.append("C:\College\Projects\Breathing Problem Classification")
from utils import CustomDataset, TrainLoopv2, EfficientNet_transform

import warnings
warnings.filterwarnings("ignore")

def finetune():
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    train_dataset = CustomDataset("Data/Processed/train_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'], transform=EfficientNet_transform)
    val_dataset = CustomDataset("Data/Processed/val_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'], transform=EfficientNet_transform)

    train_loader = DataLoader(train_dataset, 16, shuffle=True)
    val_loader = DataLoader(val_dataset, 16, shuffle=True)

    num_classes = 28
    
    model.classifier = Sequential(
        Dropout(p=0.2),
        ReLU(),
        Linear(in_features=1280, out_features=num_classes)
    )

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.005)

    TrainLoopv2(model, optimizer, criterion, train_loader, val_loader, device='cuda', num_epochs=100, early_stopping_rounds=20)

    model_path = 'Models/FinetunedEfficientNet.pth'

    save(model.state_dict(), model_path)

if __name__ == '__main__':
    finetune()