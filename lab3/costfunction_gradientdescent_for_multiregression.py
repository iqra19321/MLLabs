import numpy as np
import matplotlib.pyplot as plt

# Sample Data (Features: X1, X2 | Target: y)
X = np.array([[1, 2], [2, 3], [3, 5], [4, 6], [5, 8]])
y = np.array([5, 7, 10, 12, 15])  # Linear relation

# Initialize parameters
n_features = X.shape[1]
w = np.zeros(n_features)  # Weights
b = 0  # Bias
learning_rate = 0.01
epochs = 1000

# Cost Function (Mean Squared Error)
def compute_cost_multi(X, y, w, b):
    n = len(y)
    y_pred = np.dot(X, w) + b
    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return cost

# Gradient Descent
def gradient_descent_multi(X, y, w, b, learning_rate, epochs):
    n = len(y)
    cost_history = []

    for _ in range(epochs):
        y_pred = np.dot(X, w) + b
        dw = (-1 / n) * np.dot(X.T, (y - y_pred))  # Partial derivatives w.r.t weights
        db = (-1 / n) * np.sum(y - y_pred)        # Partial derivative w.r.t bias

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        cost_history.append(compute_cost_multi(X, y, w, b))

    return w, b, cost_history

# Train the model
w, b, cost_history = gradient_descent_multi(X, y, w, b, learning_rate, epochs)

# Predictions
y_pred = np.dot(X, w) + b

# Scatter Plot of Actual vs. Predicted Values
plt.scatter(range(len(y)), y, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='x')
plt.plot(range(len(y_pred)), y_pred, color='red', linestyle='dashed')  # Connect predictions

plt.xlabel('Data Points')
plt.ylabel('Target (y)')
plt.title('Actual vs. Predicted Values (Multi Linear Regression)')
plt.legend()
plt.show()

# Print final weights and bias
print(f"Final Weights: {w}, Bias: {b}")

