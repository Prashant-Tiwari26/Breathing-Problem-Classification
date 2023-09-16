from torch import save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear, BCEWithLogitsLoss
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

import sys
sys.path.append("C:\College\Projects\Breathing Problem Classification")
from utils import CustomDataset, TrainLoopv2, SwinV2_transform

import warnings
warnings.filterwarnings("ignore")

def finetune():
    weights = Swin_V2_T_Weights.IMAGENET1K_V1
    model = swin_v2_t(weights=weights)

    train_dataset = CustomDataset("Data/Processed/train_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'], transform=SwinV2_transform)
    val_dataset = CustomDataset("Data/Processed/val_set.csv", "Data/images", "filename", ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo'], transform=SwinV2_transform)

    train_loader = DataLoader(train_dataset, 16, shuffle=True)
    val_loader = DataLoader(val_dataset, 16, shuffle=True)

    num_classes = 28
    model.fc = Linear(model.fc.in_features, num_classes)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.005)

    TrainLoopv2(model, optimizer, criterion, train_loader, val_loader, device='cuda', num_epochs=100, early_stopping_rounds=20)

    model_path = 'Models/FinetunedSwinV2.pth'

    save(model.state_dict(), model_path)

if __name__ == '__main__':
    finetune()