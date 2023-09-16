"""
Data preprocessing script for medical metadata.

This script reads cleaned data from 'cleaned_metadata.csv', performs preprocessing operations, 
and saves the processed data to 'processed_metadata.csv'.

Dependencies: pandas, numpy, scikit-learn (StandardScaler)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def PreprocessData():
    """
    Perform preprocessing on medical metadata.
    """
    data = pd.read_csv("Data/cleaned_metadata.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    for column in data.columns:
        if data[column].dtype == 'O' and len(data[column].unique()) < 4:
            data[column].replace({'N':0, 'Y':1, 'M':0, 'F':1, 'X-ray':0, 'CT':1, 'Unclear':0}, inplace=True)
        elif data[column].dtype == 'O' and len(data[column].unique()) >= 4 and len(data[column].unique()) <= 10:
            data = pd.get_dummies(data, columns=['view'], prefix=['view'])
 
    scale = StandardScaler()
    data['offset_standardized'] = scale.fit_transform(np.array(data['offset']).reshape(-1,1))
    data.drop(['offset'], axis=1, inplace=True)
    
    data = data[['offset_standardized', 'sex', 'age', 'RT_PCR_positive', 'survival', 'intubated',
       'intubation_present', 'went_icu', 'in_icu', 'view_AP', 'view_AP Erect', 'view_AP Supine', 'view_Axial',
       'view_Coronal', 'view_L', 'view_PA', 'modality', 'filename',
       'Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19', 'Chlamydophila',
       'E.Coli', 'Fungal', 'H1N1', 'Herpes ', 'Influenza', 'Klebsiella',
       'Legionella', 'Lipoid', 'MERS-CoV', 'MRSA', 'Mycoplasma', 'No Finding',
       'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS', 'Staphylococcus',
       'Streptococcus', 'Tuberculosis', 'Unknown', 'Varicella', 'Viral',
       'todo']]

    train_set, test_set = train_test_split(data, test_size=0.2, shuffle=True, random_state=1)
    train_set, val_set = train_test_split(train_set, test_size=0.125, shuffle=True, random_state=1)

    train_set.reset_index(inplace=True, drop=True)
    test_set.reset_index(inplace=True, drop=True)
    val_set.reset_index(inplace=True, drop=True)
    
    data.to_csv("Data/Processed/processed_metadata.csv")
    train_set.to_csv("Data/Processed/train_set.csv")
    test_set.to_csv("Data/Processed/test_set.csv")
    val_set.to_csv("Data/Processed/val_set.csv")

if __name__ == '__main__':
    PreprocessData()