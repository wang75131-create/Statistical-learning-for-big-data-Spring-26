import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


BASE_DIR = Path("data/mnist_large")
PATHIM = BASE_DIR / "images.csv"
PATHLB = BASE_DIR / "labels.csv"


def run_robust_svm_experiment(n_runs=3):
    images = pd.read_csv(PATHIM, index_col=0)
    labels = pd.read_csv(PATHLB, index_col=0)
    labels = labels.rename(columns={"0": "label"})

    X = images.values
    y = labels['label'].values
    print(f"x shape: {X.shape}, label shape: {y.shape}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('svm', SVC())
    ])
    # can compare it with n_dim 2

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear']
    }

    results = []

    for i in range(n_runs):
        current_seed = 42 + i

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=current_seed
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)

        grid = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )

        print(f"running {i + 1} times (Random Seed: {current_seed})...")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append(acc)
        print(f" -> best parameter: {grid.best_params_}")
        print(f" -> accuracy: {acc:.4f}\n")

    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print(f"mean accuracy: {mean_acc:.4f}")
    print(f"Standard deviation: {std_acc:.4f}")


if __name__ == "__main__":
    run_robust_svm_experiment(n_runs=3)