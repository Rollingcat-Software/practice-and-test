# 01 — Bulk Enrollment

## Amaç

LFW veri setinden geniş bir kullanıcı havuzunu FIVUCSAS biometric-processor'a kaydederek
sonraki testlerin (FAR/FRR, search, vb.) üzerinde çalışacağı baseline veriyi oluşturmak.

## Yöntem

- LFW `lfw-deepfunneled` klasörü taranır.
- ≥5 fotoğrafı olan ilk 100 kişi seçilir (~1.400 görüntü).
- Her resim `/api/v1/enroll` endpoint'ine `tenant_id=lfw-test` ile gönderilir.
- 4 paralel worker, ~5 dakikada bitirir.
- Sonuçlar `enroll_results.csv` dosyasına yazılır (her resim için status, ok, response).

## Sonuç (son koşum)

| Metrik | Değer |
|---|---|
| Toplam görüntü | 1.405 |
| Başarılı (ok=True) | **1.342** (%95.5) |
| Başarısız | 63 |
| Süre | 285.7 sn |

Başarısızların büyük kısmı 400 hataları — LFW'de bazı resimlerde DeepFace yüz tespit edemiyor
(çok küçük yüz, çoklu yüz, aşırı poz). Bu beklenen bir veri kaybıdır (~%4-5).

## Çıktı şeması — `enroll_results.csv`

| Sütun | Açıklama |
|---|---|
| user_id | LFW klasör adı (örn. `Aaron_Eckhart`) |
| image | resim dosya adı |
| folder | LFW alt klasörü (genelde user_id ile aynı) |
| status | HTTP status (-1 = bağlantı hatası) |
| ok | True/False |
| body | response gövdesinin ilk 200 karakteri |

## Yeniden çalıştırma

```powershell
python 01_bulk_enroll\bulk_enroll_lfw.py
```

Tekrarlanan enrollment'lar `face_embeddings` tablosunda birikmez — kullanıcı başına
en fazla 5 INDIVIDUAL embedding tutulur, sonrası en düşük kaliteliyi siler (script
içinde `MAX_INDIVIDUAL_ENROLLMENTS=5`).

## Temizleme

```sql
docker exec fivucsas-postgres psql -U postgres -d identity_core_db \
  -c "DELETE FROM face_embeddings WHERE tenant_id='lfw-test';"
```
