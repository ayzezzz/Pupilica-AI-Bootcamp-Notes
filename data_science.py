import pandas as pd

# Veri setini içeriye aktar
veri = pd.read_csv("olimpiyatlar.csv")  # Olimpiyat verilerini CSV dosyasından yükle
veri_head = veri.head(15)  # İlk 15 satırı görüntüle

"""
VERİ TEMİZLİĞİ PLANLAMASI:
1. Veride eksik (NaN) değerler var: Çıkartılabilir ya da doldurulabilir.
2. 'games' sütunu gereksiz: Veriden çıkarılacak.
3. Madalya almayan sporcular: 'madalya' sütunu NaN ise bu sporcular madalya almamış demektir.
4. 'id' sütunu gereksiz görünüyor.
5. 1900 yılında bir sporcu iki takıma mı katılmış? Bu durum kontrol edilmeli.
6. Ülke kısaltmaları ('uok') ya da takım bilgisi gereksiz olabilir.
7. 1920'den önceki verilerin güvenilirliği sorgulanabilir.
"""

veri.info()  # Veri seti hakkında genel bilgi edin

# Sütun isimlerini düzenle
# Orijinal sütun adlarını daha açıklayıcı ve Türkçe olacak şekilde yeniden adlandır
veri.rename(columns={
    "ID": "id",
    "Name": "isim",
    "Sex": "cinsiyet",
    "Age": "yas",
    "Height": "boy",
    "Weight": "kilo",
    "Team": "takim",
    "NOC": "uok",  # Ulusal Olimpiyat Komitesi kısaltması
    "Games": "oyunlar",
    "Year": "yil",
    "Season": "sezon",
    "City": "sehir",
    "Sport": "spor",
    "Event": "etkinlik",
    "Medal": "madalya"
}, inplace=True)

# Gereksiz veya kullanışsız verilerin çıkarılması
veri = veri.drop(["oyunlar"], axis=1)  # 'oyunlar' sütunu çıkarıldı
veri_head = veri.head(15)  # İlk 15 satır yeniden görüntülendi
veri_duplicated = veri[veri.duplicated()]  # Birbirini tekrarlayan veriler belirlendi

# %% Eksik Veri Problemi

"""
BOY VE KİLO EKSİK VERİLERİ:
1. Eksik değerleri etkinlik bazında ortalamayla doldur.
2. Eğer etkinlik bazında ortalama hesaplanamıyorsa genel ortalamayı kullan.
MADALYA DURUMU:
- Madalya almayan sporcuların veri setinden çıkarılması planlanıyor.
"""

# Eksik veriler için genel istatistikleri incele
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.hist(veri.boy, bins=100, color="blue")
plt.title("Boy Dağılımı")

plt.figure()
plt.hist(veri.kilo, bins=100, color="green")
plt.title("Kilo Dağılımı")

describe = veri.describe()  # Sayısal değişkenlerin istatistiksel özeti

# Boy ve kilo sütunundaki eksik verilerin etkinlik bazında doldurulması
essiz_etkinlik = pd.unique(veri.etkinlik)  # Tüm benzersiz etkinliklerin listesi
veri_gecici = veri.copy()  # Orijinal veri setini korumak için geçici kopya

boy_kilo_list = ["boy", "kilo"]  # Eksik veri problemi olan sütunlar

for e in essiz_etkinlik:  # Her bir etkinlik için iterasyon
    etkinlik_filtresi = veri_gecici.etkinlik == e  # Etkinlik filtresi oluştur
    veri_filtreli = veri_gecici[etkinlik_filtresi]  # Etkinlik bazında veri filtrele

    for s in boy_kilo_list:
        ortalama = np.mean(veri_filtreli[s])  # Sütun için etkinlik ortalaması

        if not np.isnan(ortalama):
            veri_filtreli[s] = veri_filtreli[s].fillna(ortalama)
        else:
            genel_ortalama = np.mean(veri[s])  # Genel ortalama ile doldur
            veri_filtreli[s] = veri_filtreli[s].fillna(genel_ortalama)

    veri_gecici[etkinlik_filtresi] = veri_filtreli

veri = veri_gecici.copy()
veri.info()  # Düzeltilen veri setini incele

# Yaş eksik verilerinin cinsiyet ve spor bazında doldurulması
essiz_cinsiyet = pd.unique(veri.cinsiyet)  # Benzersiz cinsiyet değerleri
essiz_spor = pd.unique(veri.spor)  # Benzersiz spor türleri

for c in essiz_cinsiyet:
    for s in essiz_spor:
        cinsiyet_spor_filtresi = np.logical_and(veri.cinsiyet == c, veri.spor == s)
        veri_filtreli = veri[cinsiyet_spor_filtresi]

        ortalama = np.mean(veri_filtreli["yas"])

        if not np.isnan(ortalama):
            veri_filtreli["yas"] = veri_filtreli["yas"].fillna(ortalama)
        else:
            genel_ortalama = np.mean(veri["yas"])
            veri_filtreli["yas"] = veri_filtreli["yas"].fillna(genel_ortalama)

        veri.loc[cinsiyet_spor_filtresi, "yas"] = veri_filtreli["yas"]

veri.info()  # Eksik yaş değerleri tamamlandı

# Madalya almayan sporcuların çıkarılması
madalya_degiskeni = veri.madalya
madalya_degiskeni_filtresi = pd.isnull(madalya_degiskeni)  # Madalya NaN olanlar
veri = veri[~madalya_degiskeni_filtresi]  # NaN olan sporcular çıkarıldı

# Temizlenmiş veri setini kaydet
veri.to_csv("olimpiyatlar_temizlenmis.csv", index=False)

# %% Tek Değişkenli Veri Analizi

# Histogram çizimi için fonksiyon
def plotHistogram(degisken):
    plt.figure()
    plt.hist(veri[degisken], bins=85, color="orange")
    plt.xlabel(degisken)
    plt.show()

sayisal_degiskenler = ["yas", "boy", "kilo", "yil"]
for degisken in sayisal_degiskenler:
    plotHistogram(degisken)

# Boxplot çizimi için fonksiyon
def plotBox(degisken):
    plt.figure()
    plt.boxplot(veri[degisken])
    plt.xlabel(degisken)
    plt.show()

sayisal_degiskenler = ["yas", "boy", "kilo"]
for degisken in sayisal_degiskenler:
    plotBox(degisken)
import pandas as pd

# Veri setini içeriye aktar
veri = pd.read_csv("olimpiyatlar.csv")  # Olimpiyat verilerini CSV dosyasından yükle
veri_head = veri.head(15)  # İlk 15 satırı görüntüle

"""
VERİ TEMİZLİĞİ PLANLAMASI:
1. Veride eksik (NaN) değerler var: Çıkartılabilir ya da doldurulabilir.
2. 'games' sütunu gereksiz: Veriden çıkarılacak.
3. Madalya almayan sporcular: 'madalya' sütunu NaN ise bu sporcular madalya almamış demektir.
4. 'id' sütunu gereksiz görünüyor.
5. 1900 yılında bir sporcu iki takıma mı katılmış? Bu durum kontrol edilmeli.
6. Ülke kısaltmaları ('uok') ya da takım bilgisi gereksiz olabilir.
7. 1920'den önceki verilerin güvenilirliği sorgulanabilir.
"""

veri.info()  # Veri seti hakkında genel bilgi edin

# Sütun isimlerini düzenle
# Orijinal sütun adlarını daha açıklayıcı ve Türkçe olacak şekilde yeniden adlandır
veri.rename(columns={
    "ID": "id",
    "Name": "isim",
    "Sex": "cinsiyet",
    "Age": "yas",
    "Height": "boy",
    "Weight": "kilo",
    "Team": "takim",
    "NOC": "uok",  # Ulusal Olimpiyat Komitesi kısaltması
    "Games": "oyunlar",
    "Year": "yil",
    "Season": "sezon",
    "City": "sehir",
    "Sport": "spor",
    "Event": "etkinlik",
    "Medal": "madalya"
}, inplace=True)

# Gereksiz veya kullanışsız verilerin çıkarılması
veri = veri.drop(["oyunlar"], axis=1)  # 'oyunlar' sütunu çıkarıldı
veri_head = veri.head(15)  # İlk 15 satır yeniden görüntülendi
veri_duplicated = veri[veri.duplicated()]  # Birbirini tekrarlayan veriler belirlendi

# %% Eksik Veri Problemi

"""
BOY VE KİLO EKSİK VERİLERİ:
1. Eksik değerleri etkinlik bazında ortalamayla doldur.
2. Eğer etkinlik bazında ortalama hesaplanamıyorsa genel ortalamayı kullan.
MADALYA DURUMU:
- Madalya almayan sporcuların veri setinden çıkarılması planlanıyor.
"""

# Eksik veriler için genel istatistikleri incele
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.hist(veri.boy, bins=100, color="blue")
plt.title("Boy Dağılımı")

plt.figure()
plt.hist(veri.kilo, bins=100, color="green")
plt.title("Kilo Dağılımı")

describe = veri.describe()  # Sayısal değişkenlerin istatistiksel özeti

# Boy ve kilo sütunundaki eksik verilerin etkinlik bazında doldurulması
essiz_etkinlik = pd.unique(veri.etkinlik)  # Tüm benzersiz etkinliklerin listesi
veri_gecici = veri.copy()  # Orijinal veri setini korumak için geçici kopya

boy_kilo_list = ["boy", "kilo"]  # Eksik veri problemi olan sütunlar

for e in essiz_etkinlik:  # Her bir etkinlik için iterasyon
    etkinlik_filtresi = veri_gecici.etkinlik == e  # Etkinlik filtresi oluştur
    veri_filtreli = veri_gecici[etkinlik_filtresi]  # Etkinlik bazında veri filtrele

    for s in boy_kilo_list:
        ortalama = np.mean(veri_filtreli[s])  # Sütun için etkinlik ortalaması

        if not np.isnan(ortalama):
            veri_filtreli[s] = veri_filtreli[s].fillna(ortalama)
        else:
            genel_ortalama = np.mean(veri[s])  # Genel ortalama ile doldur
            veri_filtreli[s] = veri_filtreli[s].fillna(genel_ortalama)

    veri_gecici[etkinlik_filtresi] = veri_filtreli

veri = veri_gecici.copy()
veri.info()  # Düzeltilen veri setini incele

# Yaş eksik verilerinin cinsiyet ve spor bazında doldurulması
essiz_cinsiyet = pd.unique(veri.cinsiyet)  # Benzersiz cinsiyet değerleri
essiz_spor = pd.unique(veri.spor)  # Benzersiz spor türleri

for c in essiz_cinsiyet:
    for s in essiz_spor:
        cinsiyet_spor_filtresi = np.logical_and(veri.cinsiyet == c, veri.spor == s)
        veri_filtreli = veri[cinsiyet_spor_filtresi]

        ortalama = np.mean(veri_filtreli["yas"])

        if not np.isnan(ortalama):
            veri_filtreli["yas"] = veri_filtreli["yas"].fillna(ortalama)
        else:
            genel_ortalama = np.mean(veri["yas"])
            veri_filtreli["yas"] = veri_filtreli["yas"].fillna(genel_ortalama)

        veri.loc[cinsiyet_spor_filtresi, "yas"] = veri_filtreli["yas"]

veri.info()  # Eksik yaş değerleri tamamlandı

# Madalya almayan sporcuların çıkarılması
madalya_degiskeni = veri.madalya
madalya_degiskeni_filtresi = pd.isnull(madalya_degiskeni)  # Madalya NaN olanlar
veri = veri[~madalya_degiskeni_filtresi]  # NaN olan sporcular çıkarıldı

# Temizlenmiş veri setini kaydet
veri.to_csv("olimpiyatlar_temizlenmis.csv", index=False)

# %% Tek Değişkenli Veri Analizi

# Histogram çizimi için fonksiyon
def plotHistogram(degisken):
    plt.figure()
    plt.hist(veri[degisken], bins=85, color="orange")
    plt.xlabel(degisken)
    plt.show()

sayisal_degiskenler = ["yas", "boy", "kilo", "yil"]
for degisken in sayisal_degiskenler:
    plotHistogram(degisken)

# Boxplot çizimi için fonksiyon
def plotBox(degisken):
    plt.figure()
    plt.boxplot(veri[degisken])
    plt.xlabel(degisken)
    plt.show()

sayisal_degiskenler = ["yas", "boy", "kilo"]
for degisken in sayisal_degiskenler:
    plotBox(degisken)
