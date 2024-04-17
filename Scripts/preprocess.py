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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def PreprocessData():
    """
    Perform preprocessing on medical metadata.
    """
    data = pd.read_csv("Data/cleaned_metadata.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    for column in data.columns:
        if data[column].dtype == 'O' and len(data[column].unique()) < 4:
            data[column].replace({'N':0, 'Y':1, 'M':0, 'F':1, 'X-ray':0, 'CT':1, 'Unclear':0, np.nan:-1}, inplace=True)
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
    
    X = data[['offset_standardized', 'sex', 'age', 'RT_PCR_positive', 'survival', 'intubated',
        'intubation_present', 'went_icu', 'in_icu', 'view_AP', 'view_AP Erect', 'view_AP Supine', 'view_Axial',
        'view_Coronal', 'view_L', 'view_PA', 'modality', 'filename']]
    
    y = data[['Bacterial', 'COVID-19', 'Chlamydophila', 'E.Coli', 'Fungal', 'Herpes ', 'Influenza', 'Klebsiella',
        'Legionella', 'Lipoid', 'MERS-CoV', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis', 'Pneumonia', 'SARS',
        'Streptococcus', 'Tuberculosis', 'Varicella', 'Viral', 'todo']]
    
    mlssplit = MultilabelStratifiedShuffleSplit(test_size=0.25, random_state=42)
    train_idx, test_idx = next(mlssplit.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    mlssplit = MultilabelStratifiedShuffleSplit(test_size=0.3, random_state=42)
    train_idx, test_idx = next(mlssplit.split(X_test, y_test))
    X_test, X_val = X_test.iloc[train_idx], X_test.iloc[test_idx]
    y_test, y_val = y_test.iloc[train_idx], y_test.iloc[test_idx]

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    y_val.reset_index(inplace=True, drop=True)
    
    data.to_csv("Data/Processed/processed_metadata.csv", index=False)
    X_train.to_csv("Data/Processed/train_features.csv", index=False)
    X_test.to_csv("Data/Processed/test_features.csv", index=False)
    X_val.to_csv("Data/Processed/val_features.csv", index=False)
    y_train.to_csv("Data/Processed/train_targets.csv", index=False)
    y_test.to_csv("Data/Processed/test_targets.csv", index=False)
    y_val.to_csv("Data/Processed/val_targets.csv", index=False)

if __name__ == '__main__':
    PreprocessData()