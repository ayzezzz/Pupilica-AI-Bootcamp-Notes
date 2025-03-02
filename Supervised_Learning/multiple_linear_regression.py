import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Rastgelelik için seed belirle
np.random.seed(0)

# 2 değişkenli rastgele veri oluştur
X = np.random.rand(100, 2)
coefficient = np.array([3, 5])
y = np.dot(X, coefficient) + np.random.rand(100)  # Gürültü eklenmiş doğrusal ilişki

# 3D grafik çizimi (Gerçek veri noktaları)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y, c="b", marker="o")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
plt.title("Gerçek Veri Noktaları")

# Lineer regresyon modeli oluştur
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Model katsayıları ve kesme noktasını ekrana yazdır
print(f"Katsayılar: {lin_reg.coef_}")
print(f"Y-eksenini kestiği nokta: {lin_reg.intercept_}")

# Tahmin yüzeyi oluştur
x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)

# 3D model tahmin yüzeyi çizimi
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y, c="b", marker="o")  # Gerçek veri noktaları
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha=0.5, color="r")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
plt.title("Lineer Regresyon Modeli")

plt.show()

# ----------------- #
# **Diabetes Veri Seti ile Lineer Regresyon**
# ----------------- #

# Diabetes veri setini yükle
diabetes = load_diabetes()

# Bağımsız değişkenler (X) ve hedef değişken (y) ayrımı
X = diabetes.data
y = diabetes.target

# Eğitim ve test veri seti olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Test veri seti üzerinde modelin performansını değerlendir
test_pred = lin_reg.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

# Model doğruluk metriğini ekrana yazdır
print("Test seti RMSE:", test_rmse)
