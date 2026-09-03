from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("IRIS FLOWER CLASSIFICATION")
print("==========================")

print("Actual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:",
      round(accuracy * 100, 2), "%")

# New flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(new_flower)

print("\nNew Flower:")
print("Sepal Length = 5.1")
print("Sepal Width  = 3.5")
print("Petal Length = 1.4")
print("Petal Width  = 0.2")

if result[0] == 0:
    print("Prediction: Iris Setosa")
elif result[0] == 1:
    print("Prediction: Iris Versicolor")
else:
    print("Prediction: Iris Virginica")
