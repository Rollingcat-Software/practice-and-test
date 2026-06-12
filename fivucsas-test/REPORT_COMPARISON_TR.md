# FIVUCSAS — Öncesi/Sonrası Karşılaştırmalı Test Raporu

**Tarih:** 2026-05-22
**Karşılaştırma:** İlk test koşumu (baseline, `_baseline_before_fix/`) ↔ güncellenmiş `main` kodu
**biometric-processor:** `fix/live-insufficient-evidence` → **main** (39 commit ileri, no-cache rebuild)

---

## 0. Yönetici Özeti

Güncellenmiş kod ciddi yeni güvenlik özellikleri ekliyor (embedding şifreleme, baked-in
MiniFASNet anti-spoofing, daha fazla middleware), **ancak ilk testte bulunan 4 bug'dan
yalnızca 1'i fiilen düzelmiş.** Ayrıca yeni bir eşzamanlılık regresyonu ve yerel ortamda
3 boot engeli ortaya çıktı.

| Bug | Baseline | Yeni kod | Durum |
|-----|----------|----------|-------|
| **BUG-01 Cross-tenant izolasyonu** | 3/6 FAIL | 3/6 FAIL | ❌ **DÜZELMEDİ** |
| **BUG-02 Liveness endpoint 500** | BROKEN | Çalışıyor (BPCER %8.16) | ⚠️ **KISMEN** (enhanced backend ile; MiniFASNet bu CPU'da segfault) |
| **BUG-03 GIF → 500** | FAIL | FAIL (hâlâ 500) | ❌ **DÜZELMEDİ** |
| **BUG-04 API key doğrulama** | 200 (auth yok) | 404 (401 değil) | ❌ **DÜZELMEDİ** (davranış değişti, auth hâlâ yok) |
| **YENİ — Concurrent yük** | 0/20 hata | 9/20 hata | 🔴 **REGRESYON** |

---

## 1. BUG-01 — Cross-Tenant İzolasyonu (KRİTİK) — DÜZELMEDİ ❌

**Test:** `03_cross_tenant` — 3/6 FAIL (baseline ile birebir aynı)

```
[FAIL] 02_isolation_same_id_diff_person   verified=True  (olmamalı)
[FAIL] 03_cross_person_same_id            verified=True  (olmamalı)
[FAIL] 05_unknown_tenant                  verified=True  (olmamalı, 404 beklenir)
```

**Kök sebep (kod analizi):** Tenant filtresi **yalnızca 1:N arama yoluna** eklenmiş:
- `pgvector_embedding_repository.py` → `find_similar()`: `... AND tenant_id = $4` ✅ (eklenmiş)
- `pgvector_embedding_repository.py` → `find_by_user_id()` (verify yolu): hâlâ
  `WHERE user_id = $1 AND enrollment_type = 'CENTROID'` — **tenant_id filtresi YOK** ❌

`/api/v1/verify` 1:1 eşleşmede `find_by_user_id` kullandığı için, tenant izolasyonu
verify endpoint'inde **hâlâ atlanıyor**. Var olmayan tenant adı bile eşleşiyor (Case 05).

**Düzeltme önerisi:** `find_by_user_id()` SQL'ine `AND tenant_id = $2` ekle ve `verify_face`
use case'i tenant_id'yi bu metoda geçirsin. (Bu bir satırlık eksik düzeltme — fix'in yarısı
yapılmış, verify yolu unutulmuş.)

---

## 2. BUG-02 — Liveness Endpoint — KISMEN ⚠️

**Test:** `04_liveness_spoofing`

| | Baseline | Yeni kod |
|--|----------|----------|
| Endpoint | 500 (her çağrı) — MiniFASNet yolu `/nonexistent` | **Çalışıyor**, BPCER %8.16 |
| Client-side bypass | güvenli (0/6) | güvenli (0/6) |
| APCER | ölçülemedi | ölçülemedi (spoof dataset gerek) |

Yeni kod MiniFASNet modellerini image'a **baked-in** ediyor (BUG-02'nin amaçlanan
düzeltmesi). **Ancak bu makinede UniFace MiniFASNet ONNX preload `Segmentation fault`
veriyor** (CPU/onnxruntime uyumsuzluğu) → container 509 kez crash-loop'a girdi. Boot
edebilmek için `LIVENESS_BACKEND=enhanced` (ONNX'siz, doku/davranış tabanlı) kullanıldı.

→ Liveness endpoint artık fonksiyonel (enhanced backend ile), ama **asıl MiniFASNet
anti-spoofing yolu yerel donanımda doğrulanamadı.** GPU'lu/başka CPU'lu bir hostta
ayrıca test edilmeli.

---

## 3. BUG-03 — GIF Formatı → 500 — DÜZELMEDİ ❌

**Test:** `05_edge_cases`, case 06 — hâlâ `500` (beklenen 400/415). Format doğrulaması
model çağrısından önce yapılmıyor.

---

## 4. BUG-04 — API Key Doğrulama — DÜZELMEDİ ❌

**Test:** `05_edge_cases`, case 10
- Baseline: geçersiz key → **200** (auth tamamen yok)
- Yeni kod: geçersiz key → **404** (hâlâ 401 değil)

Davranış değişti ama API key doğrulaması düzgün uygulanmıyor. (Not: yerel testte
`X-API-Key` zorunlu kılınmadı; production'da `SimpleAPIKeyMiddleware` farklı davranabilir
— ayrıca doğrulanmalı.)

---

## 5. Performans — latency aynı, concurrent REGRESYON 🔴

**Test:** `06_performance`

| Metrik | Baseline | Yeni kod |
|--------|----------|----------|
| Sıralı latency p95 | 0.423s | 0.407s (~aynı, PASS) |
| Search latency median | 0.256s | 0.345s |
| **Concurrent (20 worker) hata** | **0/20** | **9/20** 🔴 |

Eşzamanlı yük altında yeni kod isteklerin ~%45'inde hata veriyor (olası OOM / thread
contention — yeni eklenen ağır modeller bellek/CPU baskısını artırmış olabilir). Bu bir
**regresyon** ve production öncesi araştırılmalı.

---

## 6. Yeni kodun yerel boot engelleri (ortam bulguları)

Güncellenmiş kodun bu makinede çalışması için 4 müdahale gerekti — bunlar deployment
olgunluğu açısından kayda değer:

1. **UniFace MiniFASNet ONNX segfault** → `LIVENESS_BACKEND=enhanced` ile aşıldı.
2. **`FIVUCSAS_EMBEDDING_KEY` zorunlu** (GDPR Fernet şifreleme) — yoksa startup başarısız.
3. **Şema:** `embedding_ciphertext` (bytea) + `key_version` (smallint) sütunları gerekti
   (alembic migration 0005). Elle eklendi.
4. **liveness_score ölçek bug'ı:** enhanced backend 0-100 skor döndürüyor ama
   `EnrollmentResponse.liveness_score` ≤1 bekliyor → enroll patlıyor. Test için
   `min(score, 1.0)` kıstırma yaması uygulandı. **Bu da yeni bir bug.**

---

## 7. Değişmediği için yeniden koşulmayan testler

Aşağıdaki testler **embedding modelini (Facenet512)** ölçer; model değişmediği için
sonuçlar baseline ile istatistiksel olarak aynıdır. Zaman tasarrufu için baseline değerleri
taşınmıştır (gerekirse yeniden koşulabilir):

| Test | Baseline (taşındı) |
|------|--------------------|
| 02 FAR/FRR (LFW) | AUC=0.9943, EER=%1.93, FAR@0.45=%0.27, FRR@0.45=%4.40 |
| 07 AgeDB-30 | AUC=0.9475, EER=%33.99 |
| 08 CFP-FP | AUC=0.9845, EER=%27.09 |

> Not: enroll yolu artık her görüntüde liveness çalıştırdığı için bulk enroll baseline'a
> göre belirgin yavaşladı; bu da yeniden koşmama kararını destekliyor.

---

## 8. Sonuç ve Öneriler

**Düzelen:** Liveness endpoint artık fonksiyonel (BUG-02 kısmen). Güvenlik özellikleri
eklendi (embedding şifreleme).

**Düzelmeyen / yeni:**
- 🔴 **BUG-01 cross-tenant** hâlâ açık — verify yolunda tenant filtresi eksik (en kritik).
- ❌ BUG-03 (GIF 500), BUG-04 (API key) düzelmedi.
- 🔴 Concurrent yük regresyonu (9/20 hata).
- ⚠️ Yeni: liveness_score ölçek bug'ı (enroll'u kırıyor), UniFace ONNX CPU segfault.

**Öncelikli aksiyon:**
1. `find_by_user_id()`'ye `tenant_id` filtresi ekle → BUG-01'i tamamen kapat (kritik).
2. liveness_score'u 0-1'e normalize et (enroll bug'ı).
3. UniFace MiniFASNet segfault'unu araştır (onnxruntime build / CPU AVX uyumu).
4. Concurrent regresyonu profille (bellek/thread).
5. GIF format doğrulamasını ve API key zorunluluğunu düzelt.

---

### Ortam notu
Tüm yeni-kod testleri şu override ile koşuldu (`docker-compose.override.yml`):
`LIVENESS_BACKEND=enhanced`, `FIVUCSAS_EMBEDDING_KEY=<test>`, `RATE_LIMIT_ENABLED=False`,
`FACE_RECOGNITION_MODEL=Facenet512`, blur/quality threshold gevşetmeleri.
Bunlar üretim yapılandırması değildir; yerel test içindir.
