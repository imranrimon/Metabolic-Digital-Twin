import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    data_dir = "f:/Diabetics Project/data"
    pima_path = os.path.join(data_dir, "pima-indians-diabetes-database/diabetes.csv")
    pred_path = os.path.join(data_dir, "diabetes-prediction-dataset/diabetes_prediction_dataset.csv")
    
    # 1. Load Datasets
    print("Loading Pima Dataset...")
    pima_df = pd.read_csv(pima_path)
    print("Pima Info:")
    print(pima_df.info())
    print(pima_df.head())
    
    print("\nLoading 100k Prediction Dataset...")
    pred_df = pd.read_csv(pred_path)
    print("Prediction Info:")
    print(pred_df.info())
    print(pred_df.head())
    
    # 2. Basic Stats
    print("\nPima Stats:")
    print(pima_df.describe())
    
    print("\nPrediction Stats:")
    print(pred_df.describe())
    
    # 3. Class Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Outcome', data=pima_df)
    plt.title('Pima Diabetes Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='diabetes', data=pred_df)
    plt.title('100k Dataset Diabetes Distribution')
    
    plt.tight_layout()
    plt.savefig('f:/Diabetics Project/eda_distribution.png')
    print("Distribution plot saved.")
    
    # 4. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(pima_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Pima Correlation Heatmap')
    plt.savefig('f:/Diabetics Project/pima_corr.png')
    print("Correlation heatmap saved.")

if __name__ == "__main__":
    run_eda()
