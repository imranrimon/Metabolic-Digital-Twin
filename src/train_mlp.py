from preprocess import get_processed_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE

def train_mlp(dataset_name='100k'):
    print(f"\n--- Training MLP for {dataset_name} dataset ---")
    X_train, X_test, y_train, y_test, cols = get_processed_data(dataset_name)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Simple MLP with 2 hidden layers
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    
    print("Training MLP...")
    mlp.fit(X_train_sm, y_train_sm)
    y_pred = mlp.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"MLP Results: Acc={acc:.4f}, F1={f1:.4f}")
    
    # Save results
    res = pd.DataFrame([{"Model": "MLP (Neural Net)", "Accuracy": acc, "F1-Score": f1}])
    res.to_csv(f'f:/Diabetics Project/mlp_{dataset_name}.csv', index=False)
    return mlp

if __name__ == "__main__":
    train_mlp('pima')
    train_mlp('100k')
