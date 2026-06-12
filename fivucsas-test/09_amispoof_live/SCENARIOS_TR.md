# 09 — amispoof.fivucsas.com Canlı Spoof Testi (Manuel Protokol)

## Amaç

Gerçek-zamanlı, tarayıcı-içi (webcam) anti-spoofing dedektörünü **ISO/IEC 30107-3**
metodolojisiyle test etmek. Bizim 04 testimiz tek-kare/statik görüntü üzerindeydi;
bu protokol **canlı oturum** davranışını ölçer (blink, hareket, video replay zamanla).

## Test Edilen Sistem

- **URL**: https://amispoof.fivucsas.com/  (fivucsas.com/amispoof → buraya yönlenir)
- **Mimari**: Tamamen client-side (WASM + MediaPipe), frame tarayıcıdan çıkmaz
- **Oturum**: min 5 sn (warmup ~1 sn); kararlı verdict için **≥10 sn** öneriyoruz
- **Ekranda görünen**: Verdict (LIVE/SPOOF), analizör skorları, P(spoof) kategori
  olasılıkları, liveness proof, blink/incident sayacı

## ISO 30107-3 Metrikleri (hesaplanacak)

| Metrik | Formül | Anlamı |
|---|---|---|
| **BPCER** | (SPOOF denen gerçek) / (toplam gerçek) | Gerçek kullanıcıyı reddetme oranı |
| **APCER** | (LIVE denen saldırı) / (toplam saldırı) — her PAI türü için ayrı | Saldırıyı kabul oranı |
| **ACER** | (APCER + BPCER) / 2 | Ortalama sınıflandırma hatası |

> **PAI = Presentation Attack Instrument** (saldırı aracı). ISO her tür için ayrı APCER
> ister; toplamda **en kötü (max) APCER** raporlanır (worst-case güvenlik).

## Hazırlık (materyaller)

1. **Gerçek yüz** — sen (canlı). İyi ışıklı bir oda + bir de loş oda.
2. **Basılı fotoğraf** — kendi yüzünün net bir fotoğrafını A4/foto kâğıdına bas
   (hem mat hem parlak varsa ikisi de). Yüz gerçek boyuta yakın olsun.
3. **Telefon + tablet** — aynı fotoğrafı ekranda göstermek için.
4. **Kısa video** — telefonla 15-20 sn kendi selfie videonu çek (gözünü kırp, başını
   hafif oynat). Bu, replay saldırısı için. Mümkünse tablette de oynat.

## Test Düzeni

- Her trial için sayfada **Start** → senaryoyu uygula (≥10 sn) → **Verdict** otur →
  değerleri kaydet → **Reset**.
- Kamera mesafesi: yüz/foto/ekran kareyi makul doldursun (~40-60 cm).
- Her trial'ı `results.csv`'ye bir satır olarak gir (şablon hazır).
- Mümkünse sayfanın **Record** özelliğiyle her oturumu kaydet (kanıt için).

---

## Senaryo Matrisi

### A. Bona Fide (Gerçek) — BPCER için  → beklenen: **LIVE**

| ID | Senaryo | Nasıl |
|----|---------|-------|
| G1 | Normal frontal, iyi ışık | Doğal bak, ara ara gözünü kırp |
| G2 | Kafa hareketi | Yavaşça sağa-sola çevir, başını salla |
| G3 | İfade değişimi | Gülümse, kaşını kaldır |
| G4 | Loş oda | Işığı azalt, aynı doğal davranış |
| G5 | Gözlük/aksesuar | Gözlük tak/çıkar (kullanıyorsan) |
| G6 | Hafif açı | Yüzü ~15-20° yana çevir |

### B. Basılı Fotoğraf (STATIC_IMAGE / PRINT) → beklenen: **SPOOF**

| ID | Senaryo | Nasıl |
|----|---------|-------|
| P1 | Düz parlak baskı, sabit | Fotoğrafı düz tut, sabit |
| P2 | Mat baskı, sabit | Mat kâğıt versiyonu |
| P3 | Baskı + sahte hareket | Fotoğrafı hafifçe oynat/eğ (canlı taklidi) |
| P4 | Baskı yakın çekim | Kareyi tamamen doldur |
| P5 | Göz deliği kesik baskı (opsiyonel) | Gözleri kes, arkadan kendi gözün görünsün |

### C. Ekranda Statik Foto (STATIC_IMAGE / SCREEN) → beklenen: **SPOOF**

| ID | Senaryo | Nasıl |
|----|---------|-------|
| S1 | Telefon, normal parlaklık | Fotoğrafı telefonda göster, sabit |
| S2 | Telefon, max parlaklık | Parlaklığı sonuna aç |
| S3 | Tablet (büyük ekran) | Aynı foto tablette |
| S4 | Ekran + hafif eğme | Telefonu hafifçe oynat |

### D. Video Replay (VIDEO_REPLAY) → beklenen: **SPOOF**  ⚠️ sistemin en zayıf noktası

| ID | Senaryo | Nasıl |
|----|---------|-------|
| V1 | Telefon video, normal | Selfie videonu oynat, kamerayı göster |
| V2 | Telefon video, max parlaklık | Parlaklık sonuna |
| V3 | Tablet video | Tablette oynat |
| V4 | Çok hareketli video | Bol blink + kafa hareketli video (en zor) |
| V5 | Çok sabit video | Neredeyse hareketsiz video |

**Toplam**: 6 gerçek + 14 saldırı = **20 trial**. Her biri ≥10 sn → ~25-30 dk.

---

## Her Trial İçin Kaydedilecekler (results.csv sütunları)

| Sütun | Açıklama |
|---|---|
| trial_id | G1, P1, S1, V1... |
| category | genuine / print / screen_photo / video_replay |
| expected | LIVE / SPOOF |
| verdict | Sayfadaki nihai Verdict (LIVE/SPOOF) |
| top_spoof_category | "P(spoof) by attack category" içinde en yüksek olan |
| top_spoof_prob | O kategorinin olasılığı (0-1 veya %) |
| liveness_score | Liveness proof skoru (varsa 0-100) |
| blink_count | Oturum sonu blink sayısı |
| incident_count | Oturum sonu incident sayısı |
| duration_sec | Oturum süresi |
| correct | verdict == expected ise 1, değilse 0 |
| notes | Gözlem (örn. "ekran parlaklığı yüksekken LIVE'a kaydı") |

---

## Çalıştırma Sırası

1. `results.csv`'yi aç (şablon hazır, 20 satır önceden dolu).
2. Her trial'ı sayfada uygula, gözlemlenen değerleri ilgili satıra yaz, `correct` sütununu doldur.
3. Bittiğinde metrikleri hesapla:
   ```powershell
   python 09_amispoof_live\compute_pad_metrics.py
   ```
4. Çıktı: `summary.txt` — BPCER, PAI-türü başına APCER, genel APCER (max), ACER + ISO Grade.

## ISO Grade Yorumu (kabaca)

| ACER | Grade | Yorum |
|---|---|---|
| < %5 | A | Üretim için güçlü |
| %5–%10 | B | İyi, iyileştirilebilir |
| %10–%20 | C | Sınırda — bizim 04 testimiz buradaydı (APCER %30) |
| > %20 | D | Yetersiz |

## Notlar

- Video replay (D) sonuçlarına özellikle dikkat — README'deki ground-truth'ta video
  replay "LIVE %60" yanlış sınıflanıyordu. Bu trial'lar düzelme olup olmadığını gösterir.
- Sayfa client-side olduğu için sonuçlar **senin cihazının kamerası + ışık**
  koşullarına bağlı. Aynı koşulları tüm trial'larda korumaya çalış (adil karşılaştırma).
- Bu test bizim Docker'daki biometric-processor'dan **bağımsız** — amispoof ayrı bir
  client-side demo. Yani şu an arka planda koşan testleri etkilemez.
