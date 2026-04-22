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

BASE_DIR = Path("mnist_large")
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ---------------------- Pipeline ----------------------
    models = {
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=30, random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=3))  
        ]),
        "LogReg": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.9, random_state=42)),
            ('classifier', LogisticRegression(max_iter=10000, C=0.01))  
        ]),
        "RandomForest": Pipeline([
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=300,max_depth=None, min_samples_leaf=2,max_features='log2'))  
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50, random_state=42)),
            ('classifier', SVC(C=10, kernel='rbf', random_state=42)) 
        ])
    }

    # ---------------------- main ----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in models.items():
        print(f"\nEvaluating {name}...")
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name} Cross-Validation Acc: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Test Acc: {acc:.4f}")

    print("\n=== Final Test Comparison ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_results:
        print(f"{name:.<15} {acc:.4f}")


    print("\nPlotting Final Comparison Bar Chart...")
    model_names = [item[0] for item in sorted_results]
    accuracies = [item[1] for item in sorted_results]

    plt.figure(figsize=(10, 6))
   
    bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

   
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.title('Test Accuracy Comparison of Four ML Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0, 1.05)  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_and_visualize()