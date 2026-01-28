import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from preprocess import get_processed_data
from models import AttentionResNetRisk
from ppgr_model import PPGRModel

class ExperimentOrchestrator:
    def __init__(self, results_dir='f:/Diabetics Project/results'):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.models_dir = os.path.join(results_dir, 'models')
        self.reports_dir = os.path.join(results_dir, 'reports')
        
        for d in [self.plots_dir, self.models_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.metrics_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_metrics(self):
        df = pd.DataFrame(self.metrics_list)
        df.to_csv(os.path.join(self.results_dir, 'metrics_summary.csv'), index=False)
        print(f"Metrics summary saved to {self.results_dir}/metrics_summary.csv")

    def plot_results(self, y_test, y_probs, model_name, dataset_name, ablation_type):
        suffix = f"{dataset_name}_{model_name}_{ablation_type}".replace(" ", "_")
        
        # 1. Confusion Matrix
        y_preds = (y_probs > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.plots_dir, f'cm_{suffix}.png'))
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_test, y_probs):.4f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f'ROC Curve - {model_name} ({dataset_name})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, f'roc_{suffix}.png'))
        plt.close()

    def run_sklearn_model(self, model, model_name, X_train, X_test, y_train, y_test, dataset_name, ablation_type):
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        if len(y_probs.shape) > 1 and y_probs.shape[1] == 2: y_probs = y_probs[:, 1]
        
        y_preds = (y_probs > 0.5).astype(int)
        
        metrics = {
            "dataset": dataset_name,
            "model": model_name,
            "ablation": ablation_type,
            "accuracy": accuracy_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds, zero_division=0),
            "recall": recall_score(y_test, y_preds, zero_division=0),
            "f1": f1_score(y_test, y_preds, zero_division=0),
            "auc": roc_auc_score(y_test, y_probs)
        }
        self.metrics_list.append(metrics)
        self.plot_results(y_test, y_probs, model_name, dataset_name, ablation_type)

    def run_pytorch_resnet(self, X_train, X_test, y_train, y_test, dataset_name, ablation_type):
        print(f"  Training Attention-ResNet...")
        # Simple training loop for brevity in orchestrator
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train.values).unsqueeze(-1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        
        model = AttentionResNetRisk(input_dim=X_train.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(20): # Short for global run
            model.train()
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            y_probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
            
        y_preds = (y_probs > 0.5).astype(int)
        metrics = {
            "dataset": dataset_name, "model": "Attention-ResNet", "ablation": ablation_type,
            "accuracy": accuracy_score(y_test, y_preds), "precision": precision_score(y_test, y_preds, zero_division=0),
            "recall": recall_score(y_test, y_preds, zero_division=0), "f1": f1_score(y_test, y_preds, zero_division=0),
            "auc": roc_auc_score(y_test, y_probs)
        }
        self.metrics_list.append(metrics)
        self.plot_results(y_test, y_probs, "Attention-ResNet", dataset_name, ablation_type)

    def execute(self):
        datasets = ['pima', '100k']
        
        for ds_name in datasets:
            print(f"\n>>> Starting experiments for dataset: {ds_name}")
            X_train, X_test, y_train, y_test, cols = get_processed_data(ds_name)
            
            # SMOTE
            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            configurations = [
                ("Full_Features", X_train_sm, X_test, y_train_sm, y_test)
            ]
            
            # Ablation Setup: Remove HbA1c or Glucose if present
            hba1c_idx = -1
            for i, c in enumerate(cols):
                if 'HbA1c' in c or 'hba1c' in c.lower():
                    hba1c_idx = i
                    break
            
            if hba1c_idx != -1:
                X_train_ab = np.delete(X_train_sm, hba1c_idx, axis=1)
                X_test_ab = np.delete(X_test, hba1c_idx, axis=1)
                configurations.append(("No_HbA1c", X_train_ab, X_test_ab, y_train_sm, y_test))

            for cfg_name, xtr, xte, ytr, yte in configurations:
                print(f" Configuration: {cfg_name}")
                
                # 1. Baselines
                self.run_sklearn_model(LogisticRegression(max_iter=1000), "Logistic Regression", xtr, xte, ytr, yte, ds_name, cfg_name)
                
                # Skip SVM for large datasets (too slow)
                if len(xtr) < 10000:
                    self.run_sklearn_model(SVC(probability=True), "SVM", xtr, xte, ytr, yte, ds_name, cfg_name)
                else:
                    print(f"  Skipping SVM for {ds_name} (Dataset too large)")
                    
                self.run_sklearn_model(KNeighborsClassifier(), "KNN", xtr, xte, ytr, yte, ds_name, cfg_name)
                self.run_sklearn_model(DecisionTreeClassifier(), "Decision Tree", xtr, xte, ytr, yte, ds_name, cfg_name)
                
                # 2. Proposed
                self.run_sklearn_model(RandomForestClassifier(n_estimators=100), "Random Forest", xtr, xte, ytr, yte, ds_name, cfg_name)
                self.run_sklearn_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost", xtr, xte, ytr, yte, ds_name, cfg_name)
                self.run_sklearn_model(LGBMClassifier(verbose=-1), "LightGBM", xtr, xte, ytr, yte, ds_name, cfg_name)
                
                # 3. Stacked
                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=50)),
                    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
                ]
                stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
                self.run_sklearn_model(stack, "Stacked Ensemble (RF+XGB)", xtr, xte, ytr, yte, ds_name, cfg_name)
                
                # 4. Advanced (Pytorch)
                self.run_pytorch_resnet(xtr, xte, ytr, yte, ds_name, cfg_name)
                
        self.save_metrics()
        print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator()
    orchestrator.execute()
