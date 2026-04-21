import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

PATHIM = "data/mnist_large/images.csv"
PATHLB = "data/mnist_large/labels.csv"


# Load data in the same style as the notebook.
images = pd.read_csv(PATHIM, sep=",", index_col=0)
labels = pd.read_csv(PATHLB, sep=",", index_col=0)
labels = labels.rename(columns={"0": "label"})

# Split in the same way as the shared group setup.
x_train, x_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42,
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

X = images.to_numpy()
y = labels["label"].to_numpy()
X_train = x_train.to_numpy()
X_val = x_val.to_numpy()
X_test = x_test.to_numpy()
y_train = y_train["label"].to_numpy()
y_val = y_val["label"].to_numpy()
y_test = y_test["label"].to_numpy()

print(f"Full data shape: {X.shape}")
print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")
print()

digits = np.unique(y)

for digit in digits:
    train_n = np.sum(y_train == digit)
    val_n = np.sum(y_val == digit)
    test_n = np.sum(y_test == digit)
    total_n = train_n + val_n + test_n

    print(
        f"Digit {digit}: "
        f"train={train_n}, val={val_n}, test={test_n}, "
        f"train_share={train_n / total_n:.2f}"
    )

print()


# Fit PCA on training data only.
n_components = 50
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)

print(f"Number of PCA components: {n_components}")
print(
    "Explained variance ratio (first 10 PCs):",
    np.round(pca.explained_variance_ratio_[:10], 4)
)
print()
print(
    "Cumulative explained variance:"
    f" {pca.explained_variance_ratio_.sum():.4f}"
)


# Plot cumulative explained variance.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(cumulative_variance, "b-")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot PC1 vs PC2 for the training set.
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="tab10",
    s=20,
    alpha=0.6,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Training data projected onto PC1 and PC2")
plt.colorbar(scatter, label="Digit")
plt.grid(True)
plt.tight_layout()
plt.show()


# Show the first 9 loading images.
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    loading_img = pca.components_[i].reshape(28, 28)
    ax.imshow(loading_img, cmap="gray")
    ax.set_title(f"PC{i + 1} loading")
    ax.axis("off")

plt.tight_layout()
plt.show()
