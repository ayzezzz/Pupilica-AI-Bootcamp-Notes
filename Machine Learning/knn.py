# Gerekli kütüphaneleri dahil et
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Breast Cancer veri setini yükleyerek DataFrame oluştur
cancer = load_breast_cancer()
data = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
data["target"] = cancer.target

# Özellikler (X) ve hedef değişkeni (y) belirle
X = cancer.data
y = cancer.target

# Veriyi eğitim ve test setlerine böl (70% eğitim, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Verileri standardize et: Eğitim setine fit uygula ve dönüştür, test setini dönüştür
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modeli oluştur ve eğit (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Modeli test setinde değerlendir
y_pred = knn.predict(X_test)         # Test seti tahminleri
y_pred_train = knn.predict(X_train)  # Eğitim seti tahminleri

# Doğruluk skorlarını hesapla
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)

# Sonuçları ekrana yazdır
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Training Accuracy: {train_accuracy:.2f}")
