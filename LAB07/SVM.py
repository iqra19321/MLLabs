import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Read the data and clean up column names
df = pd.read_csv(r"C:\Users\LENEVO\Desktop\ML\LAB Practicals\LAB07\transfusion.csv")
df.columns = df.columns.str.strip()

# Subset for features and target
subset_df = df[['Recency', 'Frequency', 'Monetary', 'Time', 'Target']]

# Pairplot to visualize the data
sns.pairplot(subset_df, hue='Target')
plt.show()

# Correlation heatmap
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()

# Features and target for SVM
X = df.drop('Target', axis=1)
y = df['Target']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standard scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Try different kernels with different C and gamma values
kernels = ['linear', 'poly', 'sigmoid', 'rbf']
C_values = [0.1, 1.0, 10]
gamma_values = [0.01, 0.1, 1]

# Loop over different kernel, C, and gamma values
for kernel in kernels:
    for C_value in C_values:
        for gamma_value in gamma_values:
            print(f"Training SVM with kernel={kernel}, C={C_value}, gamma={gamma_value}")
            
            # Create and train the SVM model
            model = SVC(kernel=kernel, C=C_value, gamma=gamma_value)
            model.fit(x_train_scaled, y_train)

            # Make predictions and evaluate the model
            y_pred = model.predict(x_test_scaled)

            # Print performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy Score: {accuracy}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Visualization (for 2D data)
            X_vis = df[['Recency', 'Frequency']]
            x_train_vis, x_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)
            x_train_vis_scaled = scaler.fit_transform(x_train_vis)
            x_test_vis_scaled = scaler.transform(x_test_vis)
            model.fit(x_train_vis_scaled, y_train_vis)

            # Plot decision boundary
            x_min, x_max = X_vis['Recency'].min() - 1, X_vis['Recency'].max() + 1
            y_min, y_max = X_vis['Frequency'].min() - 1, X_vis['Frequency'].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.75, cmap='coolwarm')
            plt.scatter(X_vis['Recency'], X_vis['Frequency'], c=y, edgecolors='k', marker='o', cmap='coolwarm', s=100)
            plt.xlabel('Recency')
            plt.ylabel('Frequency')
            plt.title(f'SVM Decision Boundary with {kernel} Kernel, C={C_value}, gamma={gamma_value}')
            plt.colorbar()
            plt.show()
