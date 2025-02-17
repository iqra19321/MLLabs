import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()

# Gini Index Calculation
def gini(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

# Information Gain Calculation (using Gini Index)
def information_gain(X, y, feature):
    total_gini = gini(y)

    values, counts = np.unique(X[feature], return_counts=True)

    weighted_gini = sum((counts[i] / len(y)) * gini(y[X[feature] == values[i]]) for i in range(len(values)))

    return total_gini - weighted_gini

# Find the best feature to split
def best_split(X, y):
    return max(X.columns, key=lambda feature: information_gain(X, y, feature))

class DecisionTree:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        # Base case: if only one class is left or max depth is reached
        if len(np.unique(y)) == 1 or self.depth >= self.max_depth:
            return y.mode()[0]

        best_feature = best_split(X, y)
        tree = {best_feature: {}}

        for value in X[best_feature].unique():
            subtree = DecisionTree(self.depth + 1, self.max_depth).fit(X[X[best_feature] == value], y[X[best_feature] == value])
            tree[best_feature][value] = subtree

        self.tree = tree
        return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    value = sample[feature]
    # Handle missing values by returning 'Unknown' if a value doesn't exist in the tree
    if value not in tree[feature]:
        return 'Unknown'  # Or you can use the most common class or any fallback
    return predict(tree[feature].get(value), sample)

# Select features and target
X = dataset[['Age', 'EstimatedSalary']].copy()  # Use .copy() to avoid the SettingWithCopyWarning
y = dataset['Purchased']

# Convert numerical to categorical (for simplicity)
X['Age'] = pd.cut(X['Age'], bins=3, labels=["Young", "Middle", "Old"])
X['EstimatedSalary'] = pd.cut(X['EstimatedSalary'], bins=3, labels=["Low", "Medium", "High"])

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the manual decision tree (using Gini index)
dt_manual = DecisionTree(max_depth=5)
tree_manual = dt_manual.fit(X_train, y_train)

# Prediction for the test set
y_pred_manual = [predict(tree_manual, sample) for sample in X_test.to_dict(orient='records')]

# Calculate accuracy
accuracy_manual = accuracy_score(y_test, y_pred_manual)
print(f"Manual Decision Tree Accuracy: {accuracy_manual}")

# Train Scikit-Learn Decision Tree Classifier with Gini index
dt_sklearn = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
dt_sklearn.fit(X_train, y_train)

# Prediction for the test set
y_pred_sklearn = dt_sklearn.predict(X_test)

# Calculate accuracy
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-Learn Decision Tree Accuracy: {accuracy_sklearn}")

# Scikit-Learn Decision Tree with Entropy
dt_sklearn_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_sklearn_entropy.fit(X_train, y_train)

# Prediction for the test set
y_pred_sklearn_entropy = dt_sklearn_entropy.predict(X_test)

# Calculate accuracy
accuracy_sklearn_entropy = accuracy_score(y_test, y_pred_sklearn_entropy)
print(f"Scikit-Learn Decision Tree with Entropy Accuracy: {accuracy_sklearn_entropy}")
