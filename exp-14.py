import numpy as np
from sklearn.linear_model import LinearRegression

# House data
# Area, Bedrooms, Age
X = np.array([
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 7],
    [1800, 3, 5],
    [2000, 4, 5],
    [2200, 4, 4],
    [2500, 4, 3],
    [2800, 5, 3],
    [3000, 5, 2],
    [3500, 6, 1]
])

# Price in lakhs
y = np.array([
    30, 35, 45, 55, 62,
    70, 78, 88, 95, 110
])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict existing houses
prediction = model.predict(X)

print("HOUSE PRICE PREDICTION")
print("======================")

for i in range(len(X)):
    print("Actual Price:", y[i],
          " Predicted Price:", round(prediction[i], 2))

# Predict new house
new_house = [[2000, 4, 3]]

new_price = model.predict(new_house)

print("\nNew House Details:")
print("Area = 2000 sq.ft")
print("Bedrooms = 4")
print("Age = 3 years")

print("Predicted House Price:",
      round(new_price[0], 2), "Lakhs")
