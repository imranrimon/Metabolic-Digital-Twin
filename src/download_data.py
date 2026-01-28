import kagglehub
import os
import shutil

def download_datasets():
    # Datasets to download
    datasets = [
        "uciml/pima-indians-diabetes-database",
        "iammustafatz/diabetes-prediction-dataset",
        "eniyanantony/food-suitable-for-diabetes-and-blood-pressure",
        "nandagopll/food-suitable-for-diabetes-and-blood-pressure",
        "jothammasila/diabetes-food-dataset"
    ]
    
    base_path = "f:/Diabetics Project/data"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for ds in datasets:
        print(f"Downloading {ds}...")
        path = kagglehub.dataset_download(ds)
        
        # Move files to project data folder
        ds_name = ds.split('/')[-1]
        target_dir = os.path.join(base_path, ds_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        shutil.copytree(path, target_dir)
        print(f"Dataset {ds_name} saved to {target_dir}")

if __name__ == "__main__":
    download_datasets()
