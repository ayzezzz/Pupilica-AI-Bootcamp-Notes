import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.cluster import DBSCAN

def plot_clusters(X, labels, title):
    """ Kümeleme sonuçlarını görselleştirme fonksiyonu """
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k", s=50)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.colorbar(label="Küme Etiketleri")
    plt.show()

# make_circles veri kümesi ile DBSCAN
X_circles, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)
dbscan_circles = DBSCAN(eps=0.1, min_samples=5)
labels_circles = dbscan_circles.fit_predict(X_circles)
plot_clusters(X_circles, labels_circles, "DBSCAN - Çembersel Veri Kümesi")

# make_moons veri kümesi ile DBSCAN
X_moons, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)
dbscan_moons = DBSCAN(eps=0.2, min_samples=10)
labels_moons = dbscan_moons.fit_predict(X_moons)
plot_clusters(X_moons, labels_moons, "DBSCAN - Ay Şeklinde Veri Kümesi")
