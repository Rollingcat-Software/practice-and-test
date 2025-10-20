# main.py

import numpy as np
from deepface import DeepFace

# --- GÖRSEL DOSYALARININ İSİMLERİNİ BURAYA YAZIN ---
# Proje klasörünüze attığınız dosyaların isimleriyle aynı olmalı.
img1_path = "kisi_A_1.jpg"
img2_path = "kisi_A_2.jpg"
img3_path = "kisi_B_1.jpg"

print("DeepFace Testi Başlatılıyor...")
print("-" * 30)

# --- TEST 1: verify() FONKSİYONU - Aynı Kişiyi Karşılaştırma ---
# İki fotoğrafın aynı kişiye ait olup olmadığını kontrol eder.
try:
    print(f"'{img1_path}' ve '{img2_path}' karşılaştırılıyor (Aynı kişi olmalı)...")
    result_same_person = DeepFace.verify(img1_path=img1_path, img2_path=img2_path)

    # Sonuçları daha okunaklı yazdırma
    print("Sonuç: Bu iki fotoğraf aynı kişiye mi ait? ->", result_same_person["verified"])
    print(f"Mesafe: {result_same_person['distance']:.4f} (Eşik Değer: {result_same_person['threshold']})")
    print("-" * 30)

except Exception as e:
    print(f"HATA: Fotoğraf işlenemedi. Dosya yolunu kontrol edin: {e}")
    print("-" * 30)

# --- TEST 2: verify() FONKSİYONU - Farklı Kişileri Karşılaştırma ---
try:
    print(f"'{img1_path}' ve '{img3_path}' karşılaştırılıyor (Farklı kişiler olmalı)...")
    result_different_person = DeepFace.verify(img1_path=img1_path, img2_path=img3_path)

    print("Sonuç: Bu iki fotoğraf aynı kişiye mi ait? ->", result_different_person["verified"])
    print(f"Mesafe: {result_different_person['distance']:.4f} (Eşik Değer: {result_different_person['threshold']})")
    print("-" * 30)

except Exception as e:
    print(f"HATA: Fotoğraf işlenemedi. Dosya yolunu kontrol edin: {e}")
    print("-" * 30)

# --- TEST 3: represent() FONKSİYONU - Yüz Vektörü (Embedding) Oluşturma ---
# Bir yüz fotoğrafını matematiksel bir vektöre dönüştürür.
try:
    print(f"'{img1_path}' için yüz vektörü (embedding) oluşturuluyor...")
    # Not: represent fonksiyonu bir liste içinde sonuç döner.
    embedding_obj = DeepFace.represent(img_path=img1_path)
    embedding = embedding_obj[0]["embedding"]

    print(f"Başarıyla {len(embedding)} boyutlu bir vektör oluşturuldu.")

    # Vektörün tamamını göster (düzenli format)
    print("\n=== VEKTÖRÜN TAMAMI ===")
    embedding_array = np.array(embedding)

    # Her satırda 10 değer göster
    print("\n[Düzenli Format - Her satırda 10 değer]")
    for i in range(0, len(embedding), 10):
        row = embedding[i:i + 10]
        formatted_row = ", ".join([f"{val:.6f}" for val in row])
        print(f"[{i:4d}-{min(i + 9, len(embedding) - 1):4d}]: {formatted_row}")

    # Ham vektör
    print("\n[Ham Python List Format]")
    print(embedding)

    # İstatistikler
    print(f"\n=== VEKTÖR İSTATİSTİKLERİ ===")
    print(f"Boyut: {len(embedding)}")
    print(f"Min değer: {np.min(embedding_array):.6f}")
    print(f"Max değer: {np.max(embedding_array):.6f}")
    print(f"Ortalama: {np.mean(embedding_array):.6f}")
    print(f"Standart sapma: {np.std(embedding_array):.6f}")
    print("-" * 30)

except Exception as e:
    print(f"HATA: Vektör oluşturulamadı: {e}")
    print("-" * 30)
