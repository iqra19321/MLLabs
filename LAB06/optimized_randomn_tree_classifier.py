# Scikit-learn RandomForest with GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from randomn_tree_classifier import X_train, X_test, Y_train, Y_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']  # Important for handling imbalance
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, Y_train)

print("Best Parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluate the model
metrics = {
    'Accuracy': accuracy_score(Y_test, y_pred),
    'Precision': precision_score(Y_test, y_pred),
    'Recall': recall_score(Y_test, y_pred),
    'F1 Score': f1_score(Y_test, y_pred)
}

print("Metrics:", metrics)
print("Predicted class distribution:", np.unique(y_pred, return_counts=True))

fig, axes = plt.subplots(1, 5, figsize=(25, 4), dpi=100)
for i in range(5):
    plot_tree(best_rf.estimators_[i], feature_names=X_train.columns,
              class_names=['No', 'Yes'], filled=True, ax=axes[i], max_depth=2, fontsize=7)
    axes[i].set_title(f'Tree {i+1}')
plt.tight_layout()
plt.show()
