import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline 

BASE_DIR = Path("data/mnist_large")
PATHIM = BASE_DIR / "images.csv"
PATHLB = BASE_DIR / "labels.csv"

def explore_and_visualize():
    # ---------------------- data loading and visualization ----------------------
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

    # ---------------------- split data ----------------------
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.25, stratify=y, random_state=42
    # )
    #
    # # ---------------------- Pipeline ----------------------
    # models = {
    #     "KNN": Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('pca', PCA(n_components=30, random_state=42)),
    #         ('classifier', KNeighborsClassifier(n_neighbors=3))
    #     ]),
    #     "LogReg": Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('pca', PCA(n_components=0.9, random_state=42)),
    #         ('classifier', LogisticRegression(max_iter=10000, C=0.01))
    #     ]),
    #     "RandomForest": Pipeline([
    #         ('pca', PCA(n_components=0.95, random_state=42)),
    #         ('classifier', RandomForestClassifier(n_estimators=300,max_depth=None, min_samples_leaf=2,max_features='log2'))
    #     ]),
    #     "SVM": Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('pca', PCA(n_components=50, random_state=42)),
    #         ('classifier', SVC(C=10, kernel='rbf', random_state=42))
    #     ])
    # }
    noise_levels = [0.0, 0.1, 0.5, 0.8]
    noise_std = 0.1 * 255
    n_runs = 3


    model_names = ["KNN", "LogReg", "RandomForest", "SVM"]
    model_results_mean = {name: [] for name in model_names}

    for noise_ratio in noise_levels:
        noise_pct = int(noise_ratio * 100)
        print(f"\n{'=' * 60}")
        print(f"Evaluating models with {noise_pct}% noisy features (Average over {n_runs} runs)")
        print(f"{'=' * 60}")


        current_noise_acc = {name: [] for name in model_names}

        for run_idx in range(n_runs):
            current_seed = 42 + run_idx
            print(f"\n  -> Run {run_idx + 1}/{n_runs} (Seed: {current_seed})")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=current_seed
            )

            models = {
                "KNN": Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=30, random_state=current_seed)),
                    ('classifier', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
                ]),
                "LogReg": Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.9, random_state=current_seed)),
                    ('classifier', LogisticRegression(max_iter=10000, C=0.01, n_jobs=-1))
                ]),
                "RandomForest": Pipeline([
                    ('pca', PCA(n_components=0.95, random_state=current_seed)),
                    ('classifier',
                     RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features='log2',
                                            random_state=current_seed, n_jobs=-1))
                ]),
                "SVM": Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=50, random_state=current_seed)),
                    ('classifier', SVC(C=10, kernel='rbf', random_state=current_seed))
                ])
            }

            X_train_noisy = X_train.copy().astype(float)
            X_test_noisy = X_test.copy().astype(float)

            if noise_ratio > 0:
                n_features = X_train.shape[1]
                n_noisy_features = int(n_features * noise_ratio)

                np.random.seed(current_seed)
                noisy_indices = np.random.choice(n_features, n_noisy_features, replace=False)

                X_train_noisy[:, noisy_indices] += np.random.normal(loc=0.0, scale=noise_std,
                                                                    size=(X_train.shape[0], n_noisy_features))
                X_test_noisy[:, noisy_indices] += np.random.normal(loc=0.0, scale=noise_std,
                                                                   size=(X_test.shape[0], n_noisy_features))

                X_train_noisy = np.clip(X_train_noisy, 0, 255)
                X_test_noisy = np.clip(X_test_noisy, 0, 255)

            for name, pipeline in models.items():
                pipeline.fit(X_train_noisy, y_train)
                y_pred = pipeline.predict(X_test_noisy)
                acc = accuracy_score(y_test, y_pred)
                current_noise_acc[name].append(acc)
                print(f"     {name:.<15} Acc: {acc:.4f}")

        print(f"  --- {noise_pct}% Noise Summary ---")
        for name in model_names:
            mean_acc = np.mean(current_noise_acc[name])
            model_results_mean[name].append(mean_acc)
            print(f"     {name:.<15} Mean Acc: {mean_acc:.4f}")

    plt.figure(figsize=(10, 6))

    markers = ['o', 's', '^', 'D']

    for (name, accuracies), marker in zip(model_results_mean.items(), markers):
        plt.plot(
            noise_levels,
            accuracies,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            label=name
        )

    plt.title(f'Model Robustness Comparison (Averaged over {n_runs} runs)', fontsize=14, pad=15)
    plt.xlabel('Proportion of Corrupted Features (Noise Ratio)', fontsize=12)
    plt.ylabel('Mean Test Accuracy', fontsize=12)
    plt.xticks(noise_levels)
    plt.ylim(0.85, 1.0)
    plt.legend(title='Models', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explore_and_visualize()