import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# warnings.filterwarnings("ignore", category=ConvergenceWarning)



def load_data():
    try:
        PATHIM = "data//cnd_large//images.csv"
        PATHLB = "data//cnd_large//labels.csv"
        X = pd.read_csv(PATHIM, index_col=0).values
        y = pd.read_csv(PATHLB, index_col=0)["0"].values
        print(f"Loaded real data. X shape: {X.shape}, y shape: {y.shape}")
    except FileNotFoundError:
        print("Data files not found.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=1600, n_informative=150, random_state=42)
    return X, y


def plot_selected_pixels(mask, title="Selected Pixels", ax=None):
    n_features = len(mask)
    img_size = int(math.sqrt(n_features))
    mask_2d = mask.reshape(img_size, img_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask_2d, cmap="Reds")
        ax.set_title(f"{title}\nSelected: {np.sum(mask)}")
        ax.axis("off")
        plt.show()
    else:
        ax.imshow(mask_2d, cmap="Reds")
        ax.set_title(f"{title}\nSelected: {np.sum(mask)}")
        ax.axis("off")


def exp1_filter_svm_comparison(X_train, X_test, y_train, y_test, cv):
    print("\n" + "=" * 50)
    print("Experiment 1: Filter (F-test) + SVM (Comparing Linear vs RBF)")
    print("=" * 50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("filter", SelectKBest(score_func=f_classif)),
        ("svm", SVC(random_state=42))
    ])

    param_grid = {
        'filter__k':[100, 200, 500],
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [0.1, 1, 10]
    }

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test Accuracy: {grid.score(X_test, y_test):.4f}")

    return grid.best_estimator_.named_steps["filter"].get_support()


def exp2_wrapper_svm_comparison(X_train, X_test, y_train, y_test, cv):
    print("\n" + "=" * 50)
    print("Experiment 2: Wrapper (RFE) + SVM (Comparing Linear vs RBF)")
    print("=" * 50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pre_filter", SelectKBest(score_func=f_classif, k=500)),

        ("wrapper", RFE(estimator=SVC(kernel="linear"), step=0.1)),

        ("svm", SVC(random_state=42))
    ])

    param_grid = {
        "wrapper__n_features_to_select": [50, 100, 150],

        "svm__kernel": [ "rbf"],
        "svm__C": [0.1, 1, 10]

    }

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test Accuracy: {grid.score(X_test, y_test):.4f}")

    pre_mask = grid.best_estimator_.named_steps["pre_filter"].get_support()
    wrapper_mask_small = grid.best_estimator_.named_steps["wrapper"].get_support()
    full_mask = np.zeros(len(pre_mask), dtype=bool)
    full_mask[np.where(pre_mask)[0]] = wrapper_mask_small

    return full_mask









if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mask1 = exp1_filter_svm_comparison(X_train, X_test, y_train, y_test, cv)
    mask2 = exp2_wrapper_svm_comparison(X_train, X_test, y_train, y_test, cv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_selected_pixels(mask1, "1. Filter + SVM", ax=axes[0])
    plot_selected_pixels(mask2, "2. Wrapper + SVM", ax=axes[1])
    plt.suptitle("Pure SVM Strategy: Selected Pixels Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

# C:\Users\William\AppData\Local\Programs\Python\Python313\python.exe C:\Users\William\PycharmProjects\Statistical-learning-for-big-data-Spring-26\p203.py
# Loaded real data. X shape: (10000, 4096), y shape: (10000,)
#
# ==================================================
# Experiment 1: Filter (F-test) + Linear SVM
# ==================================================
# Best Parameters: {'filter__k': 300, 'svm__C': 0.1}
# Test Accuracy: 0.7185
#
# ==================================================
# Experiment 2: Wrapper (RFE) + RBF SVM
# ==================================================
# Best Parameters: {'svm__C': 10, 'svm__gamma': 'scale', 'wrapper__n_features_to_select': 150}
# Test Accuracy: 0.8405
#
# ==================================================
# Experiment 3: Embedded (L1-LinearSVC) + Poly SVM
# ==================================================
# Best Parameters: {'embedded__estimator__C': 0.05, 'svm__C': 1}
# Test Accuracy: 0.8615
#
# ==================================================
# Stability Analysis: Using L1-LinearSVC
# ==================================================
#
# --- Part A: Statistical Stability (3 Random Runs) ---
# Average Jaccard Similarity (L1-SVM): 0.3882
#
# --- Part B: Physical Stability (Image Flipping) ---
# Pixels kept after flipping: 629 / 1412 (44.5%)
#
# 进程已结束，退出代码为 0



# C:\Users\William\AppData\Local\Programs\Python\Python313\python.exe C:\Users\William\PycharmProjects\Statistical-learning-for-big-data-Spring-26\p203.py
# Loaded real data. X shape: (10000, 4096), y shape: (10000,)
#
# ==================================================
# Experiment 1: Filter (F-test) + SVM (Comparing Linear vs RBF)
# ==================================================
# Best Parameters: {'filter__k': 500, 'svm__C': 10, 'svm__kernel': 'rbf'}
# Test Accuracy: 0.8612
#
# ==================================================
# Experiment 2: Wrapper (RFE) + SVM (Comparing Linear vs RBF)
# ==================================================
# Best Parameters: {'svm__C': 10, 'svm__kernel': 'rbf', 'wrapper__n_features_to_select': 150}
# Test Accuracy: 0.8500