import pandas as pd

file = "data/raw/dehradun_2024.csv"

df = pd.read_csv(file)

print("\n========== DATASET OVERVIEW ==========")
print("Total records:", len(df))
print("Columns:", list(df.columns))

df["period_start"] = pd.to_datetime(df["period_start"])

print("\n========== AVAILABLE PARAMETERS ==========")
print(df["parameter_name"].value_counts())

print("\n========== DATE RANGE ==========")
print("Start:", df["period_start"].min())
print("End:", df["period_start"].max())

print("\n========== UNITS ==========")
print(df.groupby("parameter_name")["unit"].unique())

# Convert long-format data into one row per day
wide = df.pivot_table(
    index="period_start",
    columns="parameter_name",
    values="mean",
    aggfunc="mean"
).reset_index()

wide.columns.name = None

wide = wide.sort_values("period_start")

print("\n========== CLEANED DATA ==========")
print(wide.head())
print("\nShape:", wide.shape)

print("\n========== MISSING VALUES ==========")
print(wide.isnull().sum())

wide.to_csv("data/raw/dehradun_cleaned.csv", index=False)

print("\nSUCCESS: data/raw/dehradun_cleaned.csv created!")
