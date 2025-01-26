import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the Insurance dataset
dataset = pd.read_csv('insurance.csv')

# Target variable
y = dataset['charges']  # 'charges' is the target


# Function to evaluate Polynomial Regression for a single predictor
def polynomial_regression(predictor, degree):
    print(f"--- Predictor: {predictor}, Polynomial Degree: {degree} ---")

    # Extract the predictor and target variable
    X = dataset[[predictor]].values  # Predictor must be 2D for PolynomialFeatures

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Transform the features into polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Fit Polynomial Regression model
    regressor = LinearRegression()
    regressor.fit(X_poly_train, y_train)

    # Predict on test data
    y_pred = regressor.predict(X_poly_test)

    # Performance metrics
    print("Coefficients:", regressor.coef_)
    print("Intercept:", regressor.intercept_)
    print('Variance score (R²):', regressor.score(X_poly_test, y_test))
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Generate a scatter plot of the predictions
    plt.style.use('fivethirtyeight')
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.scatter(X_test, y_pred, color='red', label='Predicted Data')
    plt.plot(np.sort(X_test, axis=0),
             regressor.predict(poly.transform(np.sort(X_test, axis=0))),
             color='green', linewidth=2, label='Polynomial Fit')
    plt.title(f"Polynomial Fit (Degree: {degree}) for {predictor}")
    plt.xlabel(predictor)
    plt.ylabel("Charges")
    plt.legend()
    plt.show()


# Evaluate Polynomial Regression for different predictors and degrees
predictors = ['age', 'bmi', 'children']  # Choose numerical predictors only
degrees = [1, 2, 3, 4]  # Polynomial degrees to test

for predictor in predictors:
    for degree in degrees:
        polynomial_regression(predictor, degree)
