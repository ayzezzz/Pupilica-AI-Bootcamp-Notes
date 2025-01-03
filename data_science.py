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
# %% kategorik degiskenler
veri = pd.read_csv("olimpiyatlar_temizlenmis.csv")
def plotBar(degisken, n = 5): # en cok 5 adet veriyi gorsellestir

    veri_ = veri[degisken]
    veri_sayma = veri_.value_counts() # value counts
    veri_sayma = veri_sayma[:n]
    
    plt.figure()
    plt.bar(veri_sayma.index, veri_sayma, color = "orange")
    plt.xticks(veri_sayma.index, veri_sayma.index.values)
    plt.xticks(rotation = 45)
    plt.ylabel("Frekans")
    plt.title(f"Veri Frekansi: {degisken}")
    plt.show()
    print(f"{veri_sayma}")

# dataframe icierisnde istenilen tipteki degiskenleri bul
categorical_columns = veri.select_dtypes(include = ["object"]).columns

for degisken in categorical_columns:
    plotBar(degisken, 20)
    
# %% iki degiskenli veri analizi

# cinsiyete gore boy ve kilo karsilastirmasi

erkek = veri[veri.cinsiyet == "M"]
kadin = veri[veri.cinsiyet == "F"]

plt.figure()
plt.scatter(kadin.boy, kadin.kilo, alpha = 0.8, label = "Kadin")
plt.scatter(erkek.boy, erkek.kilo, alpha = 0.1, label = "Erkek")
plt.xlabel("boy")
plt.ylabel("kilo")
plt.title("Boy ve Kilo Arasındaki İlişki")
plt.legend()

# correlation calculation
numeric_correlation = veri.loc[:, ["yas", "boy", "kilo"]].corr() 

# madalya ve yas arasindaki correlation
veri_gecici = veri.copy()
veri_gecici = pd.get_dummies(veri_gecici, columns = ["madalya"])
numeric_correlation_yas_madalya = veri_gecici.loc[:, ["yas", 'madalya_Bronze', 'madalya_Gold','madalya_Silver']].corr()

# takimlarin kazandiklari altin gumus ve vronz madalya sayilari 
# groupby
# frupladik, toplamini aldik, siraladik ve ilk 20 sini elde ettik
veri_gecici["takim"] = veri_gecici["takim"].replace({ # soviet union isimlerini rusya ile replace et degistir.
    "Soviet Union": "Russia" 
})
groupby_takim = veri_gecici[["takim", "madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["takim"], as_index = False).sum()
groupby_takim_sorted = groupby_takim.sort_values(by = "madalya_Gold", ascending = False)
groupby_takim_sorted_10 = groupby_takim_sorted[:20]

turkey = groupby_takim.query("takim == 'Turkey'")  # Sadece Türkiye'yi filtrele

# sehirlere gore kazanilan madalyalarin ortalamalari
groupby_sehir = veri_gecici[["sehir","madalya_Bronze","madalya_Silver","madalya_Gold"]].groupby(["sehir"],as_index=False).sum().sort_values(by = "madalya_Gold", ascending = False)

# cinsiyete gore
groupby_cinsiyet = veri_gecici[["cinsiyet","madalya_Bronze","madalya_Silver","madalya_Gold"]].groupby(["cinsiyet"],as_index=False).sum().sort_values(by = "madalya_Gold", ascending = False)

# %% cok degiskenli veri analizi
# pivot table

# madalya alan sporcularin cinsiyetlerine gore boy, kilo ve yas ortalamalarina bakalim
# 3 adet madalya, 2(cinsiyet)*3(boy, kilo yas)*3(mean, max, min) = 18
veri_pivot = veri.pivot_table(index = "madalya",
                              columns = "cinsiyet",
                              values = ["boy", "kilo", "yas"],
                              aggfunc = {"boy": np.mean, 
                                         "kilo": [np.median, np.max],
                                         "yas": [np.min, np.max, np.std]})

# takimlara ve cinsiyete gore alinan madalya sayilarinin toplami, max ve min degerleri

#takımlara ve cinsiyete gore alınan madalya sayıların toplamı ve maks ve min değeleri
veri_pivot_takim = veri_gecici.pivot_table(index=["takim", "sehir"], 
                                        columns = ["cinsiyet", "sezon"],
                 values=["madalya_Gold","madalya_Silver","madalya_Bronze"], 
                aggfunc={"madalya_Gold":[np.sum],
                         "madalya_Silver":[np.sum],
                         "madalya_Bronze":[np.sum]})

veri_pivot_takim["total"] = (
    veri_pivot_takim["madalya_Gold"].sum(axis =1) +
    veri_pivot_takim["madalya_Silver"].sum(axis =1) + 
    veri_pivot_takim["madalya_Bronze"].sum(axis =1))
veri_pivot_takim = veri_pivot_takim.sort_values(by = "total", ascending = False)[:100]

veri_pivot_takim.to_excel("veri_pivot_takim.xlsx")