import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC  
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

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
print("\nSetting up Pipeline (StandardScaler -> PCA fixed to 0.95 -> SVM)...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),                    
    ('pca', PCA(n_components=50, random_state=42)),
    ('svm', SVC(random_state=123))                   
])

# ==========================================
# 3. Model with cross-validation
# ==========================================
print("\nTraining SVM with Cross-Validation...")
print("Please note: SVM grid search on image data can take a long time!")

param_grid = {
    'pca__n_components': [40,50,60],
    'svm__C':[0.1,1,10], 
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma':['scale'] 
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2 
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")


results_df = pd.DataFrame(grid_search.cv_results_)


heatmap_data = results_df[['param_pca__n_components', 'param_svm__C', 'mean_test_score']].copy()


heatmap_pivot = heatmap_data.pivot(
    index='param_pca__n_components', 
    columns='param_svm__C', 
    values='mean_test_score'
)


pca_full = PCA().fit(images)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
var_to_dim = {
    var: np.argmax(cumulative_var >= var) + 1 
    for var in heatmap_pivot.index
}


ytick_labels = [f"{var:.2f}(~{var_to_dim[var]})" for var in heatmap_pivot.index]


plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    heatmap_pivot,
    annot=True,
    fmt='.3f', 
    cmap='YlGnBu_r', 
    cbar=True,
    cbar_kws={'label': 'Mean CV accuracy'},
    annot_kws={'size': 14, 'weight': 'bold'},
    linewidths=0.5,
    linecolor='white'
)


best_score = grid_search.best_score_
best_row = heatmap_pivot.index.get_loc(grid_search.best_params_['pca__n_components'])
best_col = heatmap_pivot.columns.get_loc(grid_search.best_params_['svm__C'])


circle = plt.Circle(
    (best_col + 0.5, best_row + 0.5), 
    0.25, 
    color='red', 
    fill=False, 
    linewidth=3
)
ax.add_artist(circle)


plt.title('Grid-search mean CV accuracy', fontsize=20, pad=20)
plt.xlabel('SVM Regularization Parameter (C)', fontsize=14, labelpad=15)
plt.ylabel('PCA n_components', fontsize=14, labelpad=15)
plt.yticks(
    ticks=np.arange(len(ytick_labels)) + 0.5,
    labels=ytick_labels,
    rotation=0,
    fontsize=12
)
plt.xticks(rotation=0, fontsize=12)


plt.tight_layout()
plt.show()

# ==========================================
# 4. test set evaluation
# ==========================================
print("\n=== Final Test Set Evaluation ===")
svm_pred = best_model.predict(X_test)
svm_test_acc = accuracy_score(y_test, svm_pred)

print(f"Test Accuracy: {svm_test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

cm = confusion_matrix(y_test, svm_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.show()