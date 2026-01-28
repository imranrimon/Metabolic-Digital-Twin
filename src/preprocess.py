import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_pima(filepath):
    df = pd.read_csv(filepath)
    # Handle zeros in columns where they don't make sense (Glucose, BloodPressure, etc.)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def load_and_preprocess_100k(filepath):
    df = pd.read_csv(filepath)
    
    # Label Encoding for categorical features
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    return X, y

def get_processed_data(dataset_name='100k'):
    data_dir = "f:/Diabetics Project/data"
    
    if dataset_name == 'pima':
        path = os.path.join(data_dir, "pima-indians-diabetes-database/diabetes.csv")
        X, y = load_and_preprocess_pima(path)
    else:
        path = os.path.join(data_dir, "diabetes-prediction-dataset/diabetes_prediction_dataset.csv")
        X, y = load_and_preprocess_100k(path)
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    # Test loading
    X_train, X_test, y_train, y_test, cols = get_processed_data('pima')
    print(f"Pima processed: {X_train.shape}")
    X_train, X_test, y_train, y_test, cols = get_processed_data('100k')
    print(f"100k processed: {X_train.shape}")
