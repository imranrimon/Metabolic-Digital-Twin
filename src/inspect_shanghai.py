import pandas as pd
import os

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
    t1dm_file = "f:/Diabetics Project/data/shanghai_dataset/Shanghai_T1DM/1001_0_20210730.xlsx"
    inspect_shanghai_file(t1dm_file, "f:/Diabetics Project/shanghai_inspection.txt")
    print("Inspection complete. Check shanghai_inspection.txt")
