import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

X = images.to_numpy()
y = labels["label"].to_numpy()
X_train = x_train.to_numpy()
X_test = x_test.to_numpy()
y_train = y_train["label"].to_numpy()
y_test = y_test["label"].to_numpy()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# Pipeline!
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(random_state=42)),
    ("model", LogisticRegression(max_iter=10000, random_state=42)),
])


# Tuning parameters
param_grid = {
    "pca__n_components": [0.8, 0.85, 0.9, 0.95, 0.98],
    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

# Map explained-variance ratios to concrete component counts on the
# standardized training set so the plot can use numeric PCA dimensions.
X_train_scaled_for_plot = StandardScaler().fit_transform(X_train)
pca_component_map = {}
for ratio in param_grid["pca__n_components"]:
    pca_for_plot = PCA(n_components=ratio, random_state=42)
    pca_for_plot.fit(X_train_scaled_for_plot)
    pca_component_map[ratio] = pca_for_plot.n_components_

print("Resolved PCA component counts:")
for ratio, count in pca_component_map.items():
    print(f"  {ratio} -> {count}")
print()

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print(f"Best parameter: {grid_search.best_params_}")
print()

print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
print()


# Visualize the grid-search results as a heatmap.
cv_results = pd.DataFrame(grid_search.cv_results_)
score_matrix = cv_results.pivot(
    index="param_pca__n_components",
    columns="param_model__C",
    values="mean_test_score",
).sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(score_matrix, cmap="viridis", aspect="auto")

ax.set_xticks(np.arange(len(score_matrix.columns)))
ax.set_xticklabels([str(value) for value in score_matrix.columns])
ax.set_yticks(np.arange(len(score_matrix.index)))
ax.set_yticklabels([
    f"{float(value):.2f}(~{pca_component_map[float(value)]})"
    for value in score_matrix.index
])
ax.set_xlabel("LogisticRegression C")
ax.set_ylabel("PCA n_components")
ax.set_title("Grid-search mean CV accuracy")

for row_idx, n_components in enumerate(score_matrix.index):
    for col_idx, c_value in enumerate(score_matrix.columns):
        score = score_matrix.loc[n_components, c_value]
        ax.text(
            col_idx,
            row_idx,
            f"{score:.3f}",
            ha="center",
            va="center",
            color="white" if score < score_matrix.to_numpy().mean() else "black",
        )

best_row = list(score_matrix.index).index(grid_search.best_params_["pca__n_components"])
best_col = list(score_matrix.columns).index(grid_search.best_params_["model__C"])
ax.scatter(best_col, best_row, s=180, facecolors="none", edgecolors="red", linewidths=2)

fig.colorbar(im, ax=ax, label="Mean CV accuracy")
plt.tight_layout()
plt.show()


# Evaluate the best model on test set.
best_model = grid_search.best_estimator_

y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

digits = np.unique(y)

print(f"Test accuracy: {test_accuracy:.4f}")
print()

print("Test classification report:")
print(classification_report(y_test, y_test_pred))


# Confusion matrix on test set.
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=digits,
    cmap="Blues",
    xticks_rotation="vertical",
)
plt.title("Logistic regression confusion matrix")
plt.tight_layout()
plt.show()
