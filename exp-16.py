from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = [
    ("Logistic Regression",
     LogisticRegression(max_iter=200)),

    ("K-Nearest Neighbors",
     KNeighborsClassifier(n_neighbors=5)),

    ("Decision Tree",
     DecisionTreeClassifier(random_state=42)),

    ("Random Forest",
     RandomForestClassifier(n_estimators=100,
                            random_state=42)),

    ("Naive Bayes",
     GaussianNB())
]

print("CLASSIFICATION ALGORITHM COMPARISON")
print("===================================")

best_name = ""
best_accuracy = 0

for name, model in models:

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(name, ":", round(accuracy * 100, 2), "%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name

print("\nBEST ALGORITHM")
print("==============")
print(best_name)
print("Accuracy:",
      round(best_accuracy * 100, 2), "%")
