import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

BASE_DIR = Path("data/mnist_large")
PATHIM = BASE_DIR / "images.csv"
PATHLB = BASE_DIR / "labels.csv"


def explore_and_visualize():
    # load data
    images = pd.read_csv(PATHIM, index_col=0)
    labels = pd.read_csv(PATHLB, index_col=0)
    labels = labels.rename(columns={"0": "label"})

    print(f"Shape images: {images.shape}")
    print(f"Shape labels: {labels.shape}")

    mean_intensity = images.mean(axis=1).rename("mean_intensity")
    df = pd.concat([images, labels, mean_intensity], axis=1)

    print("Plotting Mean Pixel Intensity Distribution...")
    df.pivot(columns="label", values="mean_intensity").hist(bins=40, figsize=(12, 8), sharex=True)
    plt.suptitle("Mean Pixel Intensity Distribution by Label", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    X = images.values
    y = labels['label'].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    print(f"\nPCA Component 1 explains: {explained_var[0]:.2%}")
    print(f"PCA Component 2 explains: {explained_var[1]:.2%}")
    print(f"Total variance explained by 2D projection: {sum(explained_var):.2%}")

    print("Plotting PCA Projection...")
    plt.figure(figsize=(10, 8))

    classes = np.unique(y)

    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap='tab10',
        alpha=0.6, s=15, edgecolors='none'
    )

    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=[str(c) for c in classes],
        title="Digits",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.title("PCA Projection of MNIST Subset")
    plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
    plt.ylabel(f"PC2 ({explained_var[1]:.2%})")

    plt.tight_layout()
    plt.show()

    # part1 (2)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # model
    models = {
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
        "LogReg": (LogisticRegression(max_iter=500), {"C": [0.1, 1.0]}),
        "RandomForest": (RandomForestClassifier(random_state=42), {"n_estimators": [50]}),
        "SVM": (SVC(random_state=42), {"C": [1, 10], "kernel": ["rbf"]})
    }

    results = {}

    for name, (model, param_grid) in models.items():
        print(f"\nTraining {name}...")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_pca, y_train)

        best_model = grid.best_estimator_


        y_pred = best_model.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc

        print(f"{name}: Best Params = {grid.best_params_}, Test Acc = {acc:.4f}")


    print("\n=== Final Comparison ===")
    for name, acc in results.items():
        print(f"{name:.<15} {acc:.4f}")




if __name__ == "__main__":
    explore_and_visualize()
