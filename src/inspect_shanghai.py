import pandas as pd
import os
from pathlib import Path

from metabolic_twin.config import INSPECTION_LOGS_DIR, SHANGHAI_INSPECTION_SAMPLE_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSPECTION_OUTPUT_PATH = INSPECTION_LOGS_DIR / "shanghai_inspection.txt"

def inspect_shanghai_file(file_path, output_log):
    # Use utf-8 encoding to avoid charmap errors
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write(f"Inspecting file: {file_path}\n")
        try:
            if file_path.endswith('.xls'):
                df = pd.read_excel(file_path, engine='xlrd')
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            f.write("\nColumns:\n")
            f.write(str(df.columns.tolist()) + "\n")
            f.write("\nFirst 5 rows:\n")
            f.write(df.head().to_string() + "\n")
            f.write("\nInfo:\n")
            df.info(buf=f)
            
            f.write("\nData Types:\n")
            f.write(str(df.dtypes) + "\n")
            
        except Exception as e:
            f.write(f"Error reading file: {e}\n")

if __name__ == "__main__":
    t1dm_file = SHANGHAI_INSPECTION_SAMPLE_PATH
    INSPECTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inspect_shanghai_file(t1dm_file, str(INSPECTION_OUTPUT_PATH))
    print(f"Inspection complete. Check {INSPECTION_OUTPUT_PATH}")
