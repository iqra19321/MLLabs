import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the Insurance dataset
dataset = pd.read_csv('insurance.csv')

# Separate features (X) and target variable (y)
X = dataset.iloc[:, :-1]  # All columns except 'charges' (target variable)
y = dataset.iloc[:, -1]   # 'charges' as the target variable

# Encode categorical features: 'sex', 'smoker', and 'region'
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Function to apply Multiple Linear Regression for different train-test splits
def apply_multiple_regression(test_size, random_state):
    print(f"--- Train-Test Split: test_size={test_size}, random_state={random_state} ---")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred = regressor.predict(X_test)

    # Model performance
    print("Coefficients:", regressor.coef_)
    print("Intercept:", regressor.intercept_)
    print('Variance score:', regressor.score(X_test, y_test))
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Plot residuals
    plt.style.use('fivethirtyeight')
    plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train, color="green", s=10, label='Train data')
    plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test, color="blue", s=10, label='Test data')
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
    plt.legend(loc='upper right')
    plt.title("Residual errors")
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.show()

# Experiment with different train-test splits and random states
apply_multiple_regression(test_size=0.2, random_state=0)  # 80-20 split, random_state=0
apply_multiple_regression(test_size=0.3, random_state=42)  # 70-30 split, random_state=42
apply_multiple_regression(test_size=0.25, random_state=1)  # 75-25 split, random_state=1
apply_multiple_regression(test_size=0.1, random_state=10)  # 90-10 split, random_state=10
