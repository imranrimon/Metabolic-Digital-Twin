"""
Advanced Feature Engineering for Diabetes Prediction
Implements "Grandmaster" techniques: Interaction terms, Binning, Log Transforms
"""

import pandas as pd
import numpy as np

def apply_grandmaster_features(df):
    """
    Enrich dataframe with advanced features
    """
    df = df.copy()
    
    # 1. Interaction Terms (Domain Knowledge)
    # BMI * Age: Risk compounded by age and weight
    df['BMI_Age_Interaction'] = df['bmi'] * df['age']
    
    # Glucose / HbA1c: Relationship between acute and chronic sugar levels
    # Add small epsilon to avoid division by zero
    df['Glucose_HbA1c_Ratio'] = df['blood_glucose_level'] / (df['HbA1c_level'] + 1e-5)
    
    # Age / Glucose: Metabolic resilience factor
    df['Age_Glucose_Ratio'] = df['age'] / (df['blood_glucose_level'] + 1e-5)
    
    # 2. Polynomial Features for Critical Markers
    # Curvilinear relationships often exist in biological data
    df['Glucose_Squared'] = df['blood_glucose_level'] ** 2
    df['HbA1c_Squared'] = df['HbA1c_level'] ** 2
    df['BMI_Squared'] = df['bmi'] ** 2
    
    # 3. Log Transforms for Skewed Distributions
    # Typically Insulin or Pedigree Function (not present here, but useful general practice)
    # Checking if 'age' is skewed? Not really, but let's log transform BMI which has a tail
    df['Log_BMI'] = np.log1p(df['bmi'])
    df['Log_Glucose'] = np.log1p(df['blood_glucose_level'])
    
    # 4. Binning (Categorization)
    # Medical Risk Groups
    
    # HbA1c Categories (Normal, Pre-diabetes, Diabetes)
    # <5.7 Normal, 5.7-6.4 Prediabetes, >6.4 Diabetes
    df['HbA1c_Bin'] = pd.cut(df['HbA1c_level'], 
                             bins=[-1, 5.7, 6.4, 100], 
                             labels=[0, 1, 2]).astype(int)
                             
    # Glucose Categories (Fast)
    # <100 Normal, 100-125 Prediabetes, >125 Diabetes
    df['Glucose_Bin'] = pd.cut(df['blood_glucose_level'], 
                               bins=[-1, 100, 126, 1000], 
                               labels=[0, 1, 2]).astype(int)
                               
    # BMI Categories
    # <18.5 Under, 18.5-24.9 Normal, 25-29.9 Over, >30 Obese
    df['BMI_Bin'] = pd.cut(df['bmi'], 
                           bins=[-1, 18.5, 24.9, 29.9, 100], 
                           labels=[0, 1, 2, 3]).astype(int)
                           
    # Age Groups
    # 0-18 Child, 19-35 Young Adult, 36-60 Adult, >60 Senior
    df['Age_Bin'] = pd.cut(df['age'], 
                           bins=[-1, 18, 35, 60, 120], 
                           labels=[0, 1, 2, 3]).astype(int)
                           
    return df

if __name__ == "__main__":
    # Test
    print("Testing Grandmaster Feature Engineering...")
    try:
        df = pd.read_csv('f:/Diabetics Project/data/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
        df_rich = apply_grandmaster_features(df)
        print(f"Original Cols: {df.shape[1]}")
        print(f"Enriched Cols: {df_rich.shape[1]}")
        print("New Features:", [c for c in df_rich.columns if c not in df.columns])
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
