from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_and_preprocess_data():
    """ UCI ML Repo'dan kalp hastalığı veri setini çeker, temizler ve hazırlar. """
    # Veri setini çek
    heart_disease = fetch_ucirepo(id=45)

    # Veriyi DataFrame'e çevir
    df = pd.DataFrame(data=heart_disease.data.features)
    df["target"] = heart_disease.data.targets

    # Eksik verileri temizle
    if df.isna().any().any():
        df.dropna(inplace=True)

    # Bağımsız değişkenler (X) ve hedef değişken (y)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    return X, y

def train_logistic_regression(X, y):
    """ Veriyi eğitim ve test olarak böler, lojistik regresyon modeli ile eğitir. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Lojistik Regresyon Modeli
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Model doğruluğunu hesapla
    accuracy = model.score(X_test, y_test)
    print(f"Lojistik Regresyon Modeli Doğruluğu: {accuracy:.2f}")

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    train_logistic_regression(X, y)
