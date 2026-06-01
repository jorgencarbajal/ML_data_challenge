import pandas as pd

df = pd.read_csv("data/original_data.csv", low_memory=False)

memory_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)

print(df.shape)
print(f"Memory used: {memory_gb:.2f} GB")