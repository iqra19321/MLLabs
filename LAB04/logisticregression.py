import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv('logistic regression dataset-Social_Network_Ads.csv')

# Feature selection (Drop 'User ID' as it is irrelevant)
X = dataset[['Gender', 'Age', 'EstimatedSalary']].copy()
y = dataset['Purchased']

# Encode 'Gender' column
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Task 1: Implement Logistic Regression Model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# Predictions
predictions = logistic.predict(X_test)

# Task 2: Develop Cost Function (Logistic Regression Loss)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Add Bias Column
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Initialize theta (weights) to zeros
theta = np.zeros(X_train_bias.shape[1])

# Compute Initial Cost
initial_cost = compute_cost(X_train_bias, y_train, theta)
print(f"Initial Cost: {initial_cost:.4f}")

# Task 3: Generate Confusion Matrix and Performance Metrics
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Plot Heatmap of Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
