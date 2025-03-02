import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Örnek veri kümesi oluştur (10 merkezli bloblar)
X, _ = make_blobs(n_samples=3000, centers=10, cluster_std=0.2, random_state=0)

# Veri dağılımını görselleştir
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7, edgecolors="k")
plt.title("Örnek Veri Dağılımı")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Hiyerarşik kümeleme modeli oluştur
model = AgglomerativeClustering(n_clusters=10)
cluster_labels = model.fit_predict(X)

# Dendrogram oluşturma
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(X[:1000], method="ward") 
dendrogram(linkage_matrix, no_labels=True)
plt.title("Dendrogram (Ward Metodu)")
plt.xlabel("Örnek İndeksleri")
plt.ylabel("Bağlantı Mesafesi")
plt.show()
