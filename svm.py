import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

BASE_DIR = Path("data/mnist_large")
PATHIM = BASE_DIR / "images.csv"
PATHLB = BASE_DIR / "labels.csv"


def run_robust_svm_experiment(n_runs=3, noise_levels=None):
    if noise_levels is None:
        noise_levels = [0.1, 0.5, 0.8]
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

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear']
    }

    noise_std = 0.1 * 255

    plot_noise_pcts = []
    plot_mean_accs = []
    plot_std_accs = []

    # Loop through the noise levels (10%, 50%, 80%)
    for noise_ratio in noise_levels:
        noise_pct = int(noise_ratio * 100)
        print(f"\n{'=' * 50}")
        print(f"Evaluating robustness with {noise_pct}% noisy features")
        print(f"{'=' * 50}")

        results = []

        for i in range(n_runs):
            current_seed = 42 + i

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=current_seed
            )

            np.random.seed(current_seed)
            n_features = X.shape[1]
            n_noisy_features = int(n_features * noise_ratio)

            noisy_indices = np.random.choice(n_features, n_noisy_features, replace=False)

            X_train_noisy = X_train.copy().astype(float)
            X_test_noisy = X_test.copy().astype(float)

            X_train_noisy[:, noisy_indices] += np.random.normal(loc=0.0, scale=noise_std,
                                                                size=(X_train.shape[0], n_noisy_features))
            X_test_noisy[:, noisy_indices] += np.random.normal(loc=0.0, scale=noise_std,
                                                               size=(X_test.shape[0], n_noisy_features))

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)

            grid = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
            )

            print(f"Running {i + 1}/{n_runs} (Random Seed: {current_seed})...")
            grid.fit(X_train_noisy, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test_noisy)
            acc = accuracy_score(y_test, y_pred)

            results.append(acc)
            print(f" -> best parameter: {grid.best_params_}")
            print(f" -> accuracy: {acc:.4f}\n")

        mean_acc = np.mean(results)
        std_acc = np.std(results)

        print(f"--- Summary for {noise_pct}% feature noise ---")
        print(f"Mean accuracy: {mean_acc:.4f}")
        print(f"Standard deviation: {std_acc:.4f}")

        plot_noise_pcts.append(noise_pct)
        plot_mean_accs.append(mean_acc)
        plot_std_accs.append(std_acc)

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        plot_noise_pcts,
        plot_mean_accs,
        yerr=plot_std_accs,
        marker='o',
        linestyle='-',
        color='b',
        capsize=5,
        linewidth=2,
        label='SVM Accuracy'
    )

    plt.title('SVM Performance vs. Feature Noise', fontsize=14)
    plt.xlabel('Percentage of Noisy Features (%)', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)

    plt.ylim(0.9, 1.00)
    plt.xticks(plot_noise_pcts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_robust_svm_experiment(n_runs=3, noise_levels=[0.1, 0.5, 0.8])