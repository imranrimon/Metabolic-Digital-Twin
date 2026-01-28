from preprocess import get_processed_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import os
from imblearn.over_sampling import SMOTE

def train_ensembles(dataset_name='100k'):
    print(f"\n--- Training Ensembles for {dataset_name} dataset ---")
    X_train, X_test, y_train, y_test, cols = get_processed_data(dataset_name)
    
    # Handle Class Imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else 0
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "AUC-ROC": auc
        })
        print(f"{name} Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'f:/Diabetics Project/ensemble_{dataset_name}.csv', index=False)
    print(f"Results saved to ensemble_{dataset_name}.csv")
    return results_df

if __name__ == "__main__":
    # Install imbalanced-learn first if not present
    train_ensembles('pima')
    train_ensembles('100k')
