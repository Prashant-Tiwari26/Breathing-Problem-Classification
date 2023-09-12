import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop, Normalize, Resize

transform = Compose([
    ToTensor(),
    Resize((232,232)),
    CenterCrop((224,224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset_CSVlabels(Dataset):
    """
    CustomDataset_CSVlabels is a custom PyTorch dataset class for loading image data and labels
    from a CSV file.

    Args:
        csv_file (str): The path to the CSV file containing image file names and corresponding labels.
        img_dir (str): The directory where the image files are located.
        filename_column (str): The name of the CSV column containing image file names.
        label_column (str): The name of the CSV column containing image labels.
        transform (callable, optional): A torchvision.transforms.Compose object that applies image
            transformations (default: None).

    Attributes:
        img_labels (DataFrame): A pandas DataFrame containing image file names and labels.
        img_dir (str): The directory where the image files are located.
        transform (callable): A torchvision.transforms.Compose object to apply transformations to images.

    Example:
        dataset = CustomDataset_CSVlabels(
            csv_file='labels.csv',
            img_dir='images/',
            filename_column='filename',
            label_column='label',
            transform=transform
        )

    This class allows you to create a custom dataset for loading image data and labels from a CSV file
    and applying optional image transformations during data loading.
    """
    def __init__(self,csv_file, img_dir, filename_column, label_column, transform=transform) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_labels.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.img_dir = img_dir
        self.transform = transform
        self.filename = filename_column
        self.label = label_column

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset by index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.loc[index,self.filename])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(self.img_labels.loc[index, self.label])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)