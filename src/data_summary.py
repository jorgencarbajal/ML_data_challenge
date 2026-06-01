import pandas as pd

# df = pd.read_csv("data/original_data.csv", low_memory=False)

# memory_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)

# print(df.shape)
# print(f"Memory used: {memory_gb:.2f} GB")

df = pd.read_csv("data/pass_1.csv", low_memory=False)

# Save the original Time column exactly as it was loaded
df[["Time"]].to_csv("data/time_column.csv", index=False)

# Save each unique Time value and how often it appears
time_value_counts = (
    df["Time"]
    .value_counts(dropna=False)
    .rename_axis("Time")
    .reset_index(name="Count")
)

time_value_counts.to_csv("data/time_value_counts.csv", index=False)