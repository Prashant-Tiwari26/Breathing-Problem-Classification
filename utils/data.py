import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

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