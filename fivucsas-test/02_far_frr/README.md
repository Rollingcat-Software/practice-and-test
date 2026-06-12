# 02 — FAR / FRR / EER / AUC Ölçümü

## Amaç

Face verification pipeline'ının istatistiksel doğruluğunu ölçmek:
- **FAR** (False Accept Rate) — yabancıyı yanlış kabul oranı
- **FRR** (False Reject Rate) — gerçek kullanıcıyı yanlış reddetme oranı
- **EER** — FAR=FRR olduğu nokta (tek-sayı kıyaslama metriği)
- **AUC** — ROC eğrisinin altındaki alan (model kalitesi)

## Yöntem

01'in CSV'sinden başarılı enrollment'lar yüklenir. Skorlama bias'ı önlemek için
**ayrı tenant** (`lfw-pairs`) kullanılır; her kullanıcının yalnızca **1. fotoğrafı**
enrolled, kalanlar test sorgusu olarak kullanılır.

```
Genuine pair  (label=1):  Kullanıcı U'nun K. fotoğrafı   → /verify hedef U
Imposter pair (label=0):  Kullanıcı U'nun K. fotoğrafı   → /verify hedef V (V ≠ U)
```

Her çiftin `distance` skoru toplanır; tüm threshold'lar denenerek FAR/FRR sweep yapılır.

## Sonuç (son koşum)

| Metrik | Değer | Yorum |
|---|---|---|
| Çift sayısı | 772 genuine + 4.828 imposter | Toplam 5.600 |
| **AUC** | **0.9943** | Model kalitesi üst düzey, makale benchmark seviyesi |
| **EER** | **1.93%** @ distance 0.589 | FAR=FRR olduğu denge noktası |
| **FAR @ 0.45** | **0.27%** | Üretim eşiğinde — 5.000 saldırgandan ~13'ü yanlış kabul |
| **FRR @ 0.45** | **4.40%** | 1.000 gerçek kullanıcıdan ~44'ü yanlış reddediliyor |

## Üretim eşiği (0.45) yorumu

- **Güvenlik tarafına eğilimli** — ~%0.27 FAR çok güvenli, ~%4.4 FRR ise kullanıcı
  deneyimi açısından sınırda. Bankacılık/auth için uygun, photo-tagging için sıkı.
- Daha dengeli istenirse 0.50 civarı denenebilir (FRR ~%2'ye iner, FAR %1 civarına çıkar).

## Çıktılar

| Dosya | İçerik |
|---|---|
| `pair_scores.csv` | Her çiftin label, query_user, target, distance, confidence, verified değerleri |
| `far_frr_summary.txt` | Özet: AUC, EER, threshold operating points |
| `roc.png` | ROC eğrisi (TAR vs FAR) |

## Bilinen sorun

`Summary` çıktısındaki "High-security (FAR≤0.1%)" ve "Balanced (FAR≤1%)" satırları
yanlış threshold seçiyor (en küçük distance'ı buluyor, en büyük geçerli olanı değil).
Headline metrikleri (AUC, EER, @0.45 FAR/FRR) bağımsız hesaplandığı için doğrudur.

## Yeniden çalıştırma

```powershell
python 02_far_frr\compute_far_frr.py
```

Süre: ~19 dakika (80 kullanıcı, 5.600 çift, 4 worker).

## Temizleme

```sql
docker exec fivucsas-postgres psql -U postgres -d identity_core_db \
  -c "DELETE FROM face_embeddings WHERE tenant_id='lfw-pairs';"
```
