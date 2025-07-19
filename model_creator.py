import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample training data
data = pd.DataFrame({
    "ALT": [45, 67, 38, 90, 20],
    "AST": [35, 55, 29, 80, 25],
    "ALP": [120, 133, 110, 150, 100],
    "Bilirubin": [0.8, 1.2, 0.9, 2.5, 0.6],
    "Albumin": [3.5, 3.8, 4.0, 2.9, 4.2],
    "Age": [52, 60, 47, 70, 30],
    "Target": [0, 1, 0, 1, 0]  # 0 = Normal, 1 = Abnormal
})

X = data.drop("Target", axis=1)
y = data["Target"]

model = RandomForestClassifier()
model.fit(X, y)

with open("rc_acc_68.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ rc_acc_68.pkl model saved.")
