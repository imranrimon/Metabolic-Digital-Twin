from preprocess import get_processed_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd

def evaluate_baselines(dataset_name='100k'):
    print(f"\n--- Evaluating Baselines for {dataset_name} dataset ---")
    X_train, X_test, y_train, y_test, cols = get_processed_data(dataset_name)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": SVC(kernel='linear'),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1
        })
        print(f"{name} Results: Acc={acc:.4f}, F1={f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'f:/Diabetics Project/baselines_{dataset_name}.csv', index=False)
    print(f"Results saved to baselines_{dataset_name}.csv")
    return results_df

if __name__ == "__main__":
    evaluate_baselines('pima')
    evaluate_baselines('100k')
