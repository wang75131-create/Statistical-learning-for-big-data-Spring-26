import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 0. load data
# ==========================================
PATHIM = "mnist_large/images.csv"
PATHLB = "mnist_large/labels.csv"
images = pd.read_csv(PATHIM, sep=",", index_col=0).values
labels = pd.read_csv(PATHLB, sep=",", index_col=0).values.ravel()

# ==========================================
# 1. split data
# ==========================================
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42, stratify=labels
)

# ==========================================
# 2. Pipeline setup
# ==========================================
print("\nSetting up Pipeline (PCA fixed to 0.95)...")
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)), 
    ('rf', RandomForestClassifier(random_state=123, n_jobs=-1))
])

# ==========================================
# 3. Model with cross-validation
# ==========================================
print("\nTraining Random Forest with Cross-Validation...")
param_grid = {
    'rf__n_estimators':[100, 200, 300],
    'rf__max_depth':[None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# ==========================================
# best parameters visualization
# ==========================================
results_df = pd.DataFrame(grid_search.cv_results_)


results_df['param_rf__max_depth'] = results_df['param_rf__max_depth'].fillna('None')

best_min_split = grid_search.best_params_['rf__min_samples_split']
best_max_feat = grid_search.best_params_['rf__max_features']


mask = (results_df['param_rf__min_samples_split'] == best_min_split) & \
       (results_df['param_rf__max_features'] == best_max_feat)
plot_df = results_df[mask]

plt.figure(figsize=(10, 6))
for depth in plot_df['param_rf__max_depth'].unique():
    subset = plot_df[plot_df['param_rf__max_depth'] == depth]
    subset = subset.sort_values('param_rf__n_estimators') 
    
    plt.plot(
        subset['param_rf__n_estimators'], 
        subset['mean_test_score'], 
        marker='o', 
        linewidth=2,
        label=f'max_depth: {depth}'
    )

plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Mean Cross-Validation Accuracy', fontsize=12)
plt.title(f'Random Forest Performance\n(Fixed: min_samples_split={best_min_split}, max_features={best_max_feat})', fontsize=14)
plt.legend(title='Max Depth')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 4. test set evaluation
# ==========================================
print("\n=== Final Test Set Evaluation ===")
rf_pred = best_model.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_pred)

print(f"Test Accuracy: {rf_test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()