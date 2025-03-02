import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 3D çizim için gerekli kütüphane

# Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# **2D PCA Görselleştirme**
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(8, 6))
for i, label in enumerate(target_names):
    plt.scatter(X_pca_2d[y == i, 0], X_pca_2d[y == i, 1], label=label, alpha=0.7)

plt.xlabel("1. Ana Bileşen (PC1)")
plt.ylabel("2. Ana Bileşen (PC2)")
plt.title("PCA ile 2D Iris Küme Görselleştirme")
plt.legend()
plt.grid()
plt.show()

# **3D PCA Görselleştirme**
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap="viridis", s=40, alpha=0.8)

ax.set_xlabel("1. Ana Bileşen (PC1)")
ax.set_ylabel("2. Ana Bileşen (PC2)")
ax.set_zlabel("3. Ana Bileşen (PC3)")
ax.set_title("PCA ile 3D Iris Küme Görselleştirme")
plt.show()
