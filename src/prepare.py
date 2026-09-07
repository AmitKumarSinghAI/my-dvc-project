import pandas as pd

df = pd.read_csv("data/data.csv")

print("Dataset loaded successfully!")
print(df)

df = df.dropna()

df.to_csv("data/processed.csv", index=False)

print("Processed dataset saved!")