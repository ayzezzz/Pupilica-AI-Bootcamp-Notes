import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Örnek veri kümesi oluştur (4 merkezli bloblar)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# Veri dağılımını görselleştir
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7, edgecolors="k")
plt.title("Örnek Veri Dağılımı")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# K-Means modelini oluştur ve eğit
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
kmeans.fit(X)

# Küme etiketlerini al
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Kümeleme sonuçlarını görselleştir
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7, edgecolors="k")
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Merkezler")

plt.title("K-Means Kümeleme Sonucu")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
