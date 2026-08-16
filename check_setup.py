from pathlib import Path

import pandas as pd
import sklearn


DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

print("scikit-learn version:", sklearn.__version__)
print("Dataset exists:", DATA_PATH.exists())

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())