from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df=pd.read_csv(r"C:\Users\LENEVO\Desktop\ML\LAB Practicals\LAB06\bank.csv")

labelencoders={}
for col in df.select_dtypes(include=['object']).columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    labelencoders[col]=le

X=df.drop(columns=['deposit'])    
Y=df['deposit']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
 
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape 

class CustomRandomForest:
    def __init__(self, n_estimators=5, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _select_features(self, X):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            size = int(np.sqrt(n_features))
        else:
            size = n_features
        features = np.random.choice(n_features, size, replace=False)
        return features

    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            features = self._select_features(X_sample)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append(tree)
            self.feature_indices.append(features)

    def predict(self, X):
        predictions = np.array([
            tree.predict(X[:, features]) 
            for tree, features in zip(self.trees, self.feature_indices)
        ])
        return np.round(np.mean(predictions, axis=0)).astype(int)

# Convert to NumPy arrays
X_train_np = X_train.values
X_test_np = X_test.values
Y_train_np = Y_train.values
Y_test_np = Y_test.values

# Train custom random forest
model = CustomRandomForest(n_estimators=10, max_depth=10, max_features='sqrt')
model.fit(X_train_np, Y_train_np)
y_pred = model.predict(X_test_np)

# Compute metrics
metrics = {
    'Accuracy': accuracy_score(Y_test_np, y_pred),
    'Precision': precision_score(Y_test_np, y_pred),
    'Recall': recall_score(Y_test_np, y_pred),
    'F1 Score': f1_score(Y_test_np, y_pred)
}

print("Metrics:",metrics)

fig, axes = plt.subplots(nrows=1, ncols=len(model.trees), figsize=(25, 4), dpi=100)

for i, (tree, features) in enumerate(zip(model.trees, model.feature_indices)):
    # Convert features to a list of column names
    feature_names = X_train.columns[features.tolist()]
    
    # Plot the decision tree
    plot_tree(tree, feature_names=feature_names, class_names=['No', 'Yes'],
              filled=True, ax=axes[i], max_depth=2, fontsize=7)
    
    # Set the title for each tree
    axes[i].set_title(f'Tree {i+1}')

# Adjust the layout to avoid overlap
plt.tight_layout()
plt.show()