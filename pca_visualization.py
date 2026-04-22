import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PATHIM = "data/mnist_large/images.csv"
PATHLB = "data/mnist_large/labels.csv"
IMAGE_SHAPE = (28, 28)


images = pd.read_csv(PATHIM, sep=",", index_col=0)
labels = pd.read_csv(PATHLB, sep=",", index_col=0)
labels = labels.rename(columns={"0": "label"})

x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42,
)

X = images.to_numpy()
y = labels["label"].to_numpy()
X_train = x_train.to_numpy()
X_test = x_test.to_numpy()
y_train = y_train["label"].to_numpy()
y_test = y_test["label"].to_numpy()

print(f"Full data shape: {X.shape}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print()

digits = np.unique(y)

for digit in digits:
    train_n = np.sum(y_train == digit)
    test_n = np.sum(y_test == digit)
    total_n = train_n + test_n

    print(
        f"Digit {digit}: "
        f"train={train_n}, test={test_n}, "
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
print()


# Plot cumulative explained variance.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, n_components + 1), cumulative_variance, "b-")
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
    X_train_pca[:, 3],
    c=y_train,
    cmap="tab10",
    s=20,
    alpha=0.6,
)
plt.xlabel("PC1")
plt.ylabel("PC4")
plt.title("Training data projected onto PC1 and PC4")
plt.colorbar(scatter, label="Digit")
plt.grid(True)
plt.tight_layout()
plt.show()


# Show the first 9 loading images from the unscaled PCA.
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    loading_img = pca.components_[i].reshape(IMAGE_SHAPE)
    ax.imshow(loading_img, cmap="gray")
    ax.set_title(f"PC{i + 1} loading")
    ax.axis("off")

plt.tight_layout()
plt.show()


# PCA analysis for standardized variables.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca_standardized = PCA(random_state=42)
pca_standardized.fit(X_train_scaled)

eigenvalues = pca_standardized.explained_variance_
kaiser_threshold = 1.0
selected_mask = eigenvalues > kaiser_threshold
selected_indices = np.flatnonzero(selected_mask)
selected_eigenvalues = eigenvalues[selected_mask]
selected_components = pca_standardized.components_[selected_mask]

print("Standardized-variable PCA")
print(f"Total number of available PCs: {len(eigenvalues)}")
print(f"Average eigenvalue threshold: {kaiser_threshold:.1f}")
print(f"Components with eigenvalue > 1: {len(selected_indices)}")
print(
    "Selected component indices:",
    (selected_indices + 1).tolist()
)
print()


# Scree plot with the typical selection rule threshold.
component_numbers = np.arange(1, len(eigenvalues) + 1)

plt.figure(figsize=(9, 5))
plt.plot(component_numbers, eigenvalues, color="0.6", linewidth=2.5, marker="o", markersize=2)
plt.axhline(
    y=kaiser_threshold,
    color="orange",
    linestyle="-",
    linewidth=2,
    label="Eigenvalue = 1",
)
plt.yscale("log")
plt.xlabel("Principal component")
plt.ylabel("Eigenvalue")
plt.title("Scree plot on standardized training data")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Reconstruct a few sample digits using all PCs with eigenvalue > 1.
n_demo_digits = 6
demo_indices = []
for digit in digits[:n_demo_digits]:
    demo_idx = np.flatnonzero(y_test == digit)[0]
    demo_indices.append(demo_idx)

X_test_scaled = scaler.transform(X_test)
X_test_centered = X_test_scaled

reconstruction_basis = selected_components.T
projection_scores = X_test_centered[demo_indices] @ reconstruction_basis
X_demo_reconstructed_scaled = projection_scores @ reconstruction_basis.T
X_demo_reconstructed = scaler.inverse_transform(X_demo_reconstructed_scaled)

fig, axes = plt.subplots(2, len(demo_indices), figsize=(2.2 * len(demo_indices), 5))

for col_idx, sample_idx in enumerate(demo_indices):
    original_image = X_test[sample_idx].reshape(IMAGE_SHAPE)
    reconstructed_image = X_demo_reconstructed[col_idx].reshape(IMAGE_SHAPE)
    digit_label = y_test[sample_idx]

    axes[0, col_idx].imshow(original_image, cmap="gray")
    axes[0, col_idx].set_title(f"Original: {digit_label}")
    axes[0, col_idx].axis("off")

    axes[1, col_idx].imshow(reconstructed_image, cmap="gray")
    axes[1, col_idx].set_title(f"Recon: {digit_label}")
    axes[1, col_idx].axis("off")

plt.suptitle(
    "Reconstruction using all principal components with eigenvalue > 1",
    fontsize=14,
)
plt.tight_layout()
plt.show()
