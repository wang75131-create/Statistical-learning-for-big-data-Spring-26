import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

PATHIM = "data/mnist_large/images.csv"
PATHLB = "data/mnist_large/labels.csv"

images = pd.read_csv(PATHIM, sep=",", index_col=0)
labels = pd.read_csv(PATHLB, sep=",", index_col=0)
labels = labels.rename(columns={"0": "label"})

x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.25,
    stratify=labels,
    random_state=42,
)

X_train = x_train.to_numpy()
X_test = x_test.to_numpy()
y_train = y_train["label"].to_numpy()
y_test = y_test["label"].to_numpy()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

dims = [10, 20, 25, 30, 35, 50]
k_vals = [2 * x + 1 for x in range(30)]

pipeline = Pipeline([
    ("pca", PCA(svd_solver="full", whiten=True)),
    ("knn", KNeighborsClassifier(metric="euclidean")),
])

param_grid = {
    "pca__n_components": dims,
    "knn__n_neighbors": k_vals,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2547)

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,
    return_train_score=False,
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best mean CV accuracy: {grid_search.best_score_:.4f}")
print()

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results["mean_cv_error"] = 1.0 - cv_results["mean_test_score"]

plt.figure(figsize=(10, 6))

for d in dims:
    subset = cv_results[cv_results["param_pca__n_components"] == d].sort_values("param_knn__n_neighbors")
    plt.plot(
        subset["param_knn__n_neighbors"].astype(int),
        subset["mean_cv_error"],
        marker="o",
        label=f"PCA dims={d}",
    )

plt.xlabel("k (number of neighbours)")
plt.ylabel("Mean CV Error Rate")
plt.title("KNN Mean CV Error vs k for Different PCA Dimensions")
plt.legend(title="PCA dimensions", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test accuracy: {test_accuracy:.4f}")
print()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=np.unique(y_train),
    cmap="Blues",
    xticks_rotation="vertical",
)
plt.title("KNN confusion matrix")
plt.tight_layout()
plt.show()
