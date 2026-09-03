import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Mobile features:
# RAM, Storage, Battery, Camera

X = np.array([
    [2, 32, 3000, 8],
    [2, 32, 3500, 8],
    [3, 64, 4000, 12],
    [3, 64, 4200, 13],
    [4, 64, 4500, 16],
    [4, 128, 4500, 20],
    [6, 128, 5000, 32],
    [6, 256, 5000, 48],
    [8, 128, 5000, 48],
    [8, 256, 5500, 64],
    [12, 256, 6000, 108],
    [12, 512, 6000, 108]
])

# Price categories
# 0 = Low
# 1 = Medium
# 2 = High
# 3 = Premium

y = np.array([
    0, 0, 0, 1,
    1, 1, 2, 2,
    2, 3, 3, 3
])

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X, y)

print("MOBILE PRICE PREDICTION")
print("=======================")

# New mobile
new_mobile = [[8, 256, 5000, 64]]

prediction = model.predict(new_mobile)

categories = [
    "Low Price",
    "Medium Price",
    "High Price",
    "Premium Price"
]

print("\nNew Mobile Details:")
print("RAM      : 8 GB")
print("Storage  : 256 GB")
print("Battery  : 5000 mAh")
print("Camera   : 64 MP")

print("\nPredicted Price Category:",
      categories[prediction[0]])
