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
y = dataset['charges']  # Target variable

# Define subsets of predictors
predictors_all = dataset.drop(columns=['charges'])  # All predictors
predictors_subset = dataset[['age', 'bmi', 'children', 'smoker']]  # Subset of predictors

# Function to apply Multiple Linear Regression for a given feature set
def run_regression(X, y, test_size=0.2, random_state=0):
    # Encode categorical variables if present
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    if categorical_cols:
        ct = ColumnTransformer(transformers=[
            ('encoder', OneHotEncoder(drop='first'), categorical_cols)
        ], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
    else:
        X = X.values  # Convert to numpy array if no encoding is needed

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred = regressor.predict(X_test)

    # Model performance
    print('--- Model Performance ---')
    print("Coefficients:", regressor.coef_)
    print("Intercept:", regressor.intercept_)
    print('Variance score (R²):', regressor.score(X_test, y_test))
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Residual Plot
    plt.style.use('fivethirtyeight')
    plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train, color="green", s=10, label='Train data')
    plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test, color="blue", s=10, label='Test data')
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
    plt.legend(loc='upper right')
    plt.title("Residual errors")
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.show()

# Run regression on all predictors
print("=== Model with All Predictors ===")
run_regression(predictors_all, y)

# Run regression on subset of predictors
print("\n=== Model with Subset of Predictors (age, bmi, children, smoker) ===")
run_regression(predictors_subset, y)
