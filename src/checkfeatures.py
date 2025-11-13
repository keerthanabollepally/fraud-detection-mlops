import pandas as pd

df = pd.read_csv("data/processed/train.csv")
print("Total columns: ", df.shape[1])
print("Columns: ", df.columns.tolist())
