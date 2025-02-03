import numpy as np
import matplotlib.pyplot as plt

# Sample data (X: feature, y: target)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # Perfect linear relation y = 2x

# Initialize parameters
m = 0  # Slope
b = 0  # Intercept
learning_rate = 0.01
epochs = 1000  # Iterations

# Cost function (Mean Squared Error)
def compute_cost(X, y, m, b):
    n = len(y)
    y_pred = m * X + b
    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return cost

# Gradient Descent
def gradient_descent(X, y, m, b, learning_rate, epochs):
    n = len(y)
    cost_history = []  # Store cost per epoch

    for _ in range(epochs):
        y_pred = m * X + b
        dm = (-1 / n) * np.sum(X * (y - y_pred))  # Derivative w.r.t m
        db = (-1 / n) * np.sum(y - y_pred)        # Derivative w.r.t b

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

        cost_history.append(compute_cost(X, y, m, b))

    return m, b, cost_history

# Train the model
m, b, cost_history = gradient_descent(X, y, m, b, learning_rate, epochs)

# Predictions
y_pred = m * X + b

# Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

print(f"Final Parameters: m = {m:.4f}, b = {b:.4f}")
