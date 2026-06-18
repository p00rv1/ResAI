import json
import pandas as pd

INPUT_FILE = r"C:\Users\poorv\OneDrive\Desktop\resAI\hack\candidates.jsonl"
OUTPUT_FILE = r"C:\Users\poorv\OneDrive\Desktop\resAI\hack\flat_candidates.parquet"

# Read JSONL manually
records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} candidates")

# Flatten nested dictionaries
flat_df = pd.json_normalize(
    records,
    sep="_"
)

print("Shape:", flat_df.shape)
print("\nFirst 20 columns:")
print(flat_df.columns[:20].tolist())

# Save parquet
flat_df.to_parquet(
    OUTPUT_FILE,
    engine="pyarrow",
    compression="snappy",
    index=False
)

print(f"\nSaved to {OUTPUT_FILE}")