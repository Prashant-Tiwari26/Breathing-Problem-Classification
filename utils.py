import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torchvision.transforms import v2, InterpolationMode

RegNet_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.55),
    v2.RandomVerticalFlip(0.55),
    v2.RandomRotation(45, InterpolationMode.BILINEAR),
    v2.Resize((232,232), interpolation=InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop((224,224)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EfficientNet_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.55),
    v2.RandomVerticalFlip(0.55),
    v2.RandomRotation(45, InterpolationMode.BILINEAR),
    v2.Resize((384,384), interpolation=InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop((384,384)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SwinV2_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.55),
    v2.RandomVerticalFlip(0.55),
    v2.RandomRotation(45, InterpolationMode.BILINEAR),
    v2.Resize((260,260), interpolation=InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop((256,256)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class ImagesOnlyDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading images with associated labels.

    Args:
        filenames (pd.DataFrame): DataFrame containing filenames.
        targets (pd.DataFrame): DataFrame containing target labels.
        img_dir (str): Directory containing the images.
        transform (v2._container.Compose): Composition of image transformations.

    Attributes:
        img_labels (pd.DataFrame): DataFrame containing target labels.
        filenames (pd.DataFrame): DataFrame containing filenames.
        img_dir (str): Directory containing the images.
        transform (v2._container.Compose): Composition of image transformations.
    """

    def __init__(self, filenames: pd.DataFrame, targets: pd.DataFrame, img_dir: str, resize: int, crop: int, train: bool = True) -> None:
        """
        Initializes ImagesOnlyDataset with given filenames, targets, image directory, and transformation.

        Args:
            filenames (pd.DataFrame): DataFrame containing filenames.
            targets (pd.DataFrame): DataFrame containing target labels.
            img_dir (str): Directory containing the images.
            transform (v2._container.Compose): Composition of image transformations.
        """
        super().__init__()
        self.img_labels = targets
        self.filenames = filenames
        self.img_dir = img_dir
        self.transform = self.get_transform(resize, crop, train)

    @staticmethod
    def get_transform(resize: int, crop: int, train: bool = True):
        """
        Generates a torchvision transform pipeline for image preprocessing.

        Args:
            resize (int): The size to which images will be resized.
            crop (int): The size of the central crop to be taken after resizing.
            train (bool, optional): Flag indicating whether the transform is for training or not. Defaults to True.

        Returns:
            torchvision.transforms.Compose: A composed transform pipeline.
        """
        transform = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC, antialias=True)
        ]
        if train:
            transform.extend([
                v2.RandomHorizontalFlip(0.55),
                v2.RandomVerticalFlip(0.55),
                v2.RandomRotation(45, InterpolationMode.BILINEAR)
            ])
        transform.extend([
            v2.CenterCrop(crop),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return v2.Compose(transform)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Retrieves an image and its associated label from the dataset.

        Args:
            index (int): Index to retrieve the sample.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        img_path = os.path.join(self.img_dir, self.filenames.iloc[index])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(self.img_labels.iloc[index])

        image = self.transform(image)

        return (image, y_label)

def train_loop(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    scheduler,
    fig_save_path: str,
    num_epochs:int=20,
    batch_loss: int = 1,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    device:str='cpu',
):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    val_accuracies = []

    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\n---------------------\nEpoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, (batch, label) in enumerate(train_dataloader):
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, label.float())
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            val_true_labels = []
            val_pred_labels = []
            for batch, label in val_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label.float())
                validation_loss += loss

                outputs = torch.round(torch.sigmoid(outputs))
                val_true_labels.extend(label.cpu().numpy())
                val_pred_labels.extend(outputs.cpu().numpy())

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            val_true_labels = np.array(val_true_labels)
            val_pred_labels = np.array(val_pred_labels)
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            val_accuracies.append(val_accuracy)

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))

        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

        try:
            scheduler.step(validation_loss)
        except:
            scheduler.step()

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)
    val_accuracies = np.array(val_accuracies)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=range(len(total_train_loss)), y=total_train_loss, ax=ax1, label='Training Loss')
    sns.lineplot(x=range(len(total_val_loss)), y=total_val_loss, ax=ax1, label='Validation Loss')

    ax2 = ax1.twinx()

    sns.lineplot(x=range(len(val_accuracies)), y=val_accuracies, ax=ax2, label='Validation Accuracy', color='g')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Loss and accuracy during training")
    plt.subplots_adjust(wspace=0.3)
    ax1.grid(True, linestyle='--')  
    ax2.grid(False)
    plt.xticks(range(len(total_train_loss)))
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_loop_acc(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    scheduler,
    fig_save_path: str,
    num_epochs:int=20,
    batch_loss: int = 1,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    device:str='cpu',
):
    model.to(device)
    best_val_acc = 0
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    val_accuracies = []

    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\n---------------------\nEpoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, (batch, label) in enumerate(train_dataloader):
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, label.float())
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            val_true_labels = []
            val_pred_labels = []
            for batch, label in val_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label.float())
                validation_loss += loss

                outputs = torch.round(torch.sigmoid(outputs))
                val_true_labels.extend(label.cpu().numpy())
                val_pred_labels.extend(outputs.cpu().numpy())

            val_true_labels = np.array(val_true_labels)
            val_pred_labels = np.array(val_pred_labels)
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            val_accuracies.append(val_accuracy)

            if val_accuracy >  best_val_acc:
                best_val_acc = val_accuracy
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            print("Current Validation Accuracy = {:.2f}%".format(val_accuracy*100))
            print("Best Validation Accuracy = {:.2f}%".format(best_val_acc*100))
            print(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))

        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

        try:
            scheduler.step(validation_loss, mode='max')
        except:
            scheduler.step()

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)
    val_accuracies = np.array(val_accuracies)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=range(len(total_train_loss)), y=total_train_loss, ax=ax1, label='Training Loss')
    sns.lineplot(x=range(len(total_val_loss)), y=total_val_loss, ax=ax1, label='Validation Loss')

    ax2 = ax1.twinx()

    sns.lineplot(x=range(len(val_accuracies)), y=val_accuracies, ax=ax2, label='Validation Accuracy', color='g')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Loss and accuracy during training")
    plt.subplots_adjust(wspace=0.3)
    ax1.grid(True, linestyle='--')  
    ax2.grid(False)
    plt.xticks(range(len(total_train_loss)))
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.show()