import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Rastgele non-lineer veri oluştur
np.random.seed(42)  # Tekrar üretilebilirlik için seed belirle
X = np.random.rand(100, 1) * 4  # [0, 4] arasında rastgele x değerleri
y = 2 + 3 * X**2 + np.random.randn(100, 1)  # Non-lineer ilişki (2 + 3x^2 + gürültü)

# Veri noktalarını görselleştir
plt.scatter(X, y, label="Gerçek Veri", color="blue", alpha=0.6)

# 2. dereceden polinom özellikleri oluştur
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Polinom regresyon modelini oluştur ve eğit
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Modelin tahmin ettiği değerleri görselleştir
X_fit = np.linspace(0, 4, 100).reshape(-1, 1)  # 0 ile 4 arasında x değerleri
X_fit_poly = poly_features.transform(X_fit)  # X değerlerine polinom dönüşümü uygula
y_fit = poly_reg.predict(X_fit_poly)  # Model ile tahmin yap

# Polinom regresyon çizgisini çiz
plt.plot(X_fit, y_fit, color='red', label="Polinom Regresyon", linewidth=2)

# Grafik ayarları
plt.xlabel("X Değeri")
plt.ylabel("Y Değeri")
plt.title("Polinom Regresyon Modeli (2. Derece)")
plt.legend()
plt.grid(True)
plt.show()
