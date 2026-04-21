import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATHIM = "data/mnist_large/images.csv"
PATHLB = "data/mnist_large/labels.csv"


# Load data in the same style as the notebook examples.
images = pd.read_csv(PATHIM, sep=",", index_col=0)
labels = pd.read_csv(PATHLB, sep=",", index_col=0)
labels = labels.rename(columns={"0": "label"})

print(f"Image shape: {images.shape}")
print(f"Label shape: {labels.shape}")
print()

print("Missing values in images:")
print(images.isna().sum().sum())
print()

print("Missing values in labels:")
print(labels.isna().sum().sum())
print()

print("Label counts:")
print(labels["label"].value_counts().sort_index())
print()


# Join labels to make simple summaries easier.
df = images.join(labels)
df["mean_intensity"] = images.mean(axis=1)

print("Mean intensity summary:")
print(df["mean_intensity"].describe())
print()

print("Mean intensity by label:")
print(df.groupby("label")["mean_intensity"].mean())
print()


# Histogram of average image brightness for each class.
df.pivot(columns="label", values="mean_intensity").hist(
    bins=30,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)
plt.suptitle("Mean intensity by digit")
plt.tight_layout()
plt.show()


# Convert to numpy for image plotting, as in the notebook.
images_np = np.array(images)
labels_np = np.array(labels["label"])


# Show a few random images.
np.random.seed(42)
n_samples = 9
idx = np.random.choice(images_np.shape[0], size=n_samples, replace=False)
images_sub = images_np[idx]
labels_sub = labels_np[idx]

fig, axes = plt.subplots(3, 3, figsize=(7, 7))

for i, ax in enumerate(axes.ravel()):
    img = images_sub[i].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {labels_sub[i]}")
    ax.axis("off")

plt.suptitle("Random sample images")
plt.tight_layout()
plt.show()


# Show one example image for each digit.
unique_digits = sorted(labels["label"].unique())
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, digit in zip(axes.ravel(), unique_digits):
    first_index = labels.index[labels["label"] == digit][0]
    img = images.loc[first_index].to_numpy().reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Digit: {digit}")
    ax.axis("off")

plt.suptitle("One example from each class")
plt.tight_layout()
plt.show()
