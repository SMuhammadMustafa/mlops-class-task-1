# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model to a file using pickle
with open("iris_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
    