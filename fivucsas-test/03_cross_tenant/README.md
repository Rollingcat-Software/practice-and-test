# 03 — Cross-Tenant Isolation Test

## Amaç

Multi-tenant mimarinin **veri sızıntısı (data leakage)** açısından sağlamlığını
doğrulamak. Bir tenant'ta kayıtlı yüz embedding'i, başka tenant'ın isteklerinde
ASLA match etmemeli — aynı `user_id` her iki tenant'ta da varsa bile.

Bu, FIVUCSAS gibi multi-tenant SaaS bir auth platformu için **kritik güvenlik testi**dir.
Bir banka tenant'ının müşterisi, başka bir e-ticaret tenant'ının verify endpoint'inden
asla erişilebilir olmamalı.

## Yöntem

3 farklı LFW kişisi seçilir:

```
P1 = Aaron_Eckhart
P2 = Adam_Sandler
P3 = Adrien_Brody
```

İki tenant kurulur, **kasıtlı olarak çakışan user_id'lerle**:

```
Tenant "ct-test-A":              Tenant "ct-test-B":
  user001 -> P1 (Aaron)            user001 -> P3 (Adrien)   <-- aynı id, FARKLI kişi!
  user002 -> P2 (Adam)             (yok)
```

Sonra 6 test case `/verify` ile çalıştırılır.

## Test Case'leri

| # | İstek | Hedef | Beklenen | Neyi kanıtlar |
|---|---|---|---|---|
| 01 | P1 fotoğrafı | A/user001 | verified=true | **Kontrol pozitifi** — sistem çalışıyor |
| 02 | P1 fotoğrafı | B/user001 | verified=false | **İzolasyon** — A'nın user001 kaydı B'ye sızmıyor |
| 03 | P3 fotoğrafı | A/user001 | verified=false | İzolasyon (ters yön) |
| 04 | P1 fotoğrafı | A/ghost | 404 | Var olmayan kullanıcı |
| 05 | P1 fotoğrafı | ct-ghost/user001 | 404 | Var olmayan tenant |
| 06 | P1 fotoğrafı | A/user002 | verified=false | Aynı tenant, farklı kullanıcı |

## En önemli case'ler

- **Case 02 ve 03**: Asıl sızıntı testleri. Eğer bunlar PASS değilse, sistem multi-tenant
  güvenliği sağlamıyor demektir → kritik güvenlik açığı.
- Diğerleri sanity check ve genel davranış doğrulaması.

## Çıktılar

| Dosya | İçerik |
|---|---|
| `results.csv` | 6 case'in detaylı sonucu (status, verified, distance, raw response) |
| `summary.txt` | Toplam PASS/FAIL özeti, satır satır verdict |

## Çalıştırma

```powershell
python 03_cross_tenant\test_cross_tenant.py
```

Süre: ~30 saniye. Test idempotent — başında ve sonunda kendi enrollment'larını
temizler, defalarca çalıştırılabilir.

## Yorumlama

- **6/6 PASS** → cross-tenant izolasyon sağlam, raporda bu sonucu kullanabilirsin.
- **Case 02 veya 03 FAIL** → derhal kod incelemesi gerekiyor. `face_embeddings` tablosunda
  `tenant_id` filtresi eksik veya `IS NULL` durumu yanlış işleniyor olabilir.
- **04/05 FAIL** → sızıntı yok ama hata davranışı temiz değil (200 dönüyor 404 yerine).
  Güvenlik problemi değil ama API tutarlılığı problemi.
