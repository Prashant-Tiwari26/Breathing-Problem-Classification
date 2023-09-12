"""
Script for cleaning and preprocessing metadata from a CSV file.

This script reads data from 'metadata.csv', performs data cleaning operations, and
saves the cleaned data to 'cleaned_metadata.csv'.

Dependencies: pandas, numpy
"""

import pandas as pd
import numpy as np

def CleanData():
    """
    Read data from 'metadata.csv', clean it, and save the cleaned data to 'cleaned_metadata.csv'.
    """
    data = pd.read_csv("Data/metadata.csv")

    finds_array = []
    for i, disease in enumerate(data['finding']):
        finds_array.append(disease)

    splits = []

    for disease in finds_array:
        if '/' in disease:
            splits.extend(disease.split('/'))
        else:
            splits.append(disease)

    unique_values = np.unique(splits)

    for column in unique_values:
        data[column] = 0

    for i in range(0,len(data)):
        diseases = data.loc[i, 'finding']
        diseases = diseases.split('/')

        for disease in diseases:
            data.loc[i, disease] = 1

    data.drop(['license', 'patientid', 'folder', 'url', 'other_notes', 'clinical_notes', 'doi', 'finding', 'Unnamed: 29', 'location', 'neutrophil_count', 'lymphocyte_count', 'extubated', 'temperature', 'pO2_saturation', 'leukocyte_count', 'date', 'needed_supplemental_O2'], axis=1, inplace=True)

    data.drop(data[data['filename'].str.endswith(".gz")].index, inplace=True)

    data.to_csv('Data/cleaned_metadata.csv')

if __name__ == '__main__':
    CleanData()