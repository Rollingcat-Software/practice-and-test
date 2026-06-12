# FIVUCSAS Face Verification — Test Suite

Bu klasör, FIVUCSAS face verification pipeline'ını sistematik olarak test eden adımlardan oluşur.
Her adım kendi alt klasöründe; betik (script), girdi/çıktı verisi, sonuç özetleri ve açıklama
README'si birlikte saklanır.

## Klasör Yapısı

```
fivucsas-test/
├── README.md                        ← bu dosya (genel rehber)
├── 01_bulk_enroll/                  ← LFW veri setinden 1342 yüz enrolled
│   ├── bulk_enroll_lfw.py
│   ├── enroll_results.csv
│   └── README.md
├── 02_far_frr/                      ← FAR/FRR/EER/AUC ölçümü
│   ├── compute_far_frr.py
│   ├── pair_scores.csv
│   ├── far_frr_summary.txt
│   ├── roc.png
│   └── README.md
├── 03_cross_tenant/                 ← Multi-tenant izolasyon testi
│   ├── test_cross_tenant.py
│   ├── results.csv
│   ├── summary.txt
│   └── README.md
├── 04_liveness_spoofing/            ← Liveness endpoint + bypass tests
│   ├── test_liveness_spoofing.py
│   ├── liveness_results.csv
│   ├── bypass_results.csv
│   └── summary.txt
├── 05_edge_cases/                   ← Input validation (format, size, no face)
│   ├── test_edge_cases.py
│   ├── edge_results.csv
│   └── summary.txt
└── 06_performance/                  ← Latency p50/p95, concurrent load, rate limit
    ├── test_performance.py
    ├── latency_results.csv
    └── summary.txt
```

## Ortam

- Docker Compose ile FIVUCSAS biometric-processor çalışır durumda olmalı
  (`http://localhost:8001`).
- LFW veri seti: `C:\Users\hp\Documents\GitHub\Dataset\lfw-deepfunneled`
- Gerekli paketler: `pip install requests pillow`

## Çalıştırma Sırası

```powershell
python 01_bulk_enroll\bulk_enroll_lfw.py      # bir kez yeterli
python 02_far_frr\compute_far_frr.py
python 03_cross_tenant\test_cross_tenant.py
python 04_liveness_spoofing\test_liveness_spoofing.py
python 05_edge_cases\test_edge_cases.py
python 06_performance\test_performance.py
```

## Sonuçlar Özeti

| # | Adım | Sonuç | Notlar |
|---|---|---|---|
| 01 | Bulk Enrollment | 1342 enrolled / 63 fail | Normal — bazı LFW fotoğrafları yüz içermiyor |
| 02 | FAR/FRR | **AUC=0.9943, EER=1.93%** | Üretim threshold=0.45: FAR=0.27%, FRR=4.40% |
| 03 | Cross-Tenant | **3/6 FAIL — KRİTİK** | tenant_id tamamen ignore ediliyor |
| 04 | Liveness | **BROKEN** | MiniFASNet model yolu hatalı (`/nonexistent`) |
| 05 | Edge Cases | **8/10 PASS** | 2 bug bulundu |
| 06 | Performance | **p95=0.423s PASS** | Concurrent'ta yavaşlama var |

## Bulunan Kritik Güvenlik Açıkları

### BUG-01 — Multi-Tenant İzolasyonu Tamamen Yok (KRİTİK)
**Test:** `03_cross_tenant`

`tenant_id` parametresi `/verify` endpoint'inde hiç uygulanmıyor.
- Tenant B'de kayıtlı kullanıcıya, Tenant A'dan gelen sorgu eşleşiyor
- **Var olmayan bir tenant** adı bile `verified=true` döndürüyor (case 05)
- Tüm embedding'ler global aranıyor — kiracı ayrımı yok

**Etki:** Bir kullanıcı, başka bir tenant'ın verilerine erişebilir.
**Düzeltme:** `/verify` sorgusuna `WHERE tenant_id = :tenant_id` filtresi ekle.

### BUG-02 — Liveness Endpoint Tamamen Çalışmıyor (YÜKSEK)
**Test:** `04_liveness_spoofing`

`POST /api/v1/liveness` her istekte 500 döndürüyor.
Hata: `Permission denied: '/nonexistent'` — MiniFASNet model dosyası bulunamıyor.

**Etki:** Anti-spoofing tamamen devre dışı. Basılı fotoğraf veya ekran replay ile sistemin bypass edilebileceği doğrulanamaz.
**Düzeltme:** `MINIFAS_MODEL_PATH` ortam değişkenini doğru bir yola ayarla.

### BUG-03 — GIF Formatı 500 Döndürüyor (ORTA)
**Test:** `05_edge_cases`, case 06

GIF dosyası gönderildiğinde 400/415 yerine 500 Internal Server Error döndürüyor.
**Düzeltme:** Format doğrulamasını model çağrısından önce yap.

### BUG-04 — API Anahtarı Doğrulaması Yok (ORTA)
**Test:** `05_edge_cases`, case 10

`X-Api-Key: INVALID_KEY_XYZ_000` gönderildiğinde 401 yerine 200 döndürüyor.
API anahtarı doğrulaması ya yapılmıyor ya da devre dışı.

## Pozitif Bulgular

- **FAR/FRR** : AUC=0.9943 — model kalitesi çok iyi.
- **Client-side bypass** : `liveness_passed=true` gibi alanlar server tarafında doğru şekilde görmezden geliniyor.
- **Latency** : p95=0.423s — hedef olan 1.5s'nin çok altında.
- **Büyük dosya** : 12MB girişte doğru şekilde 413 döndürüyor.
- **Corrupt input** : Bozuk JPEG ve text file doğru şekilde reddediliyor.

## Sıradaki Adımlar

- [ ] BUG-01'i düzelt ve cross-tenant testini yeniden çalıştır
- [ ] BUG-02'yi düzelt (model yolu), liveness testini yeniden çalıştır
- [ ] CelebA-Spoof indirip APCER ölçümü yap
- [ ] Rate limiting production ortamda aktif mi kontrol et (RATE_LIMIT_ENABLED)
- [ ] Concurrent load altında bellek tüketimini izle (94% uyarısı)
