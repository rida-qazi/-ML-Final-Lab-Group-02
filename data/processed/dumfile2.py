import pandas as pd

df = pd.read_csv("data/processed/ravdess_features.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nEmotion counts:")
print(df["emotion"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())