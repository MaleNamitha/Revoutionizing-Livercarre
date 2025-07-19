import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Sample data (replace with real patient dataset)
data = pd.DataFrame({
    "ALT": [45, 67, 38],
    "AST": [35, 55, 29],
    "ALP": [120, 133, 110],
    "Bilirubin": [0.8, 1.2, 0.9],
    "Albumin": [3.5, 3.8, 4.0],
    "Age": [52, 60, 47]
})

scaler = StandardScaler()
scaler.fit(data)

with open("normalizer.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ normalizer.pkl created.")
