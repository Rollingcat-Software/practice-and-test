# FIVUCSAS Yüz Doğrulama — Tam Test Raporu (Türkçe)

**Tarih:** 2026-04-27
**Hazırlayan:** QA / Biyometrik Test Ekibi
**Sistem:** FIVUCSAS Biyometrik İşlemci — Yüz Tanıma ve Canlılık Tespiti Pipeline'ı
**API Adresi:** http://localhost:8001/api/v1
**Test Veri Seti:** LFW Deep-Funneled (100 kimlik, 1.405 görüntü)

---

## Yönetici Özeti

FIVUCSAS yüz doğrulama pipeline'ı üzerinde; kayıt doğruluğu, biyometrik metrikler (FAR/FRR), çok kiracılı güvenlik izolasyonu, canlılık ve sahte saldırı tespiti, girdi doğrulama ve performans alanlarını kapsayan tam kapsamlı bir test gerçekleştirilmiştir. Yüz tanıma modeli mükemmel bir seviyede performans göstermektedir (AUC 0,9943). Ancak **dört güvenlik açığı tespit edilmiş** olup bunlardan ikisi kritik seviyede olup üretim ortamına geçişten önce giderilmesi zorunludur.

| Alan | Durum | Detay |
|---|---|---|
| Yüz Tanıma Doğruluğu | ✅ Mükemmel | AUC 0,9943, EER %1,93 |
| Girdi Doğrulama | ✅ Büyük Ölçüde İyi | 10 testten 8'i geçti |
| Performans (tek istek) | ✅ Geçti | p95 = 0,423s, hedef ≤ 1,5s |
| Çok Kiracılı İzolasyon | ❌ Kritik | tenant_id tamamen görmezden geliniyor |
| Canlılık Endpoint'i | ❌ Kritik | Bozuk — tüm çağrılarda 500 hatası |
| Sahte Saldırı Tespiti (verify) | ❌ Yüksek Risk | Sahte görüntülerin %88'i kabul ediliyor |
| Puzzle Canlılık Bypass | ❌ Kritik | Sahte metadata ile bypass mümkün |
| API Kimlik Doğrulama | ❌ Orta | Geçersiz API anahtarı 200 ile kabul ediliyor |

---

## 1. Kayıt (Enrollment) — Toplu Veri Yükleme

**Ne test edildi:** LFW veri setinden 100 kimliğe ait 1.405 görüntü, paralel çalışan bir script aracılığıyla `POST /api/v1/enroll` endpoint'i üzerinden sisteme kaydedildi.

**Sonuçlar:**

| Metrik | Değer |
|---|---|
| Toplam gönderilen görüntü | 1.405 |
| Başarıyla kaydedilen | 1.342 (%95,5) |
| Başarısız (yüz tespit edilemedi) | 63 (%4,5) |
| Kayıt edilen kullanıcı sayısı | 100 |
| Embedding boyutu | 512 (Facenet512) |

**Tespitler:**
- %4,5 başarısızlık oranı, net yüz içermeyen veya grup fotoğrafı barındıran LFW görüntüleri için beklenen aralıktadır.
- Kalite skorları 70 ile 87 arasında değişmiş, tümü yapılandırılmış 40 eşiğinin üzerinde kalmıştır.
- **Dikkat — kalite eşiği düşük ayarlanmış:** Mevcut `QUALITY_THRESHOLD=40` değeri oldukça toleranslıdır. Bina girişi veya sınav sistemi gibi güvenliğin kritik olduğu ortamlarda bu eşiğin **60–70'e yükseltilmesi** önerilir. Daha sıkı bir kalite filtresi daha kaliteli embedding üretir; bu da doğrudan daha düşük FAR ve FRR değerleriyle sonuçlanır. 60'ın altında kalan görüntüler çoğunlukla zayıf aydınlatma, hafif bulanıklık veya açılı çekim gibi sorunlar taşır — bunların tümü sonraki eşleştirme adımlarında hata riskini artırır.
- Kayıt sırasında `liveness_score` alanı her zaman 1,0 döndürmüştür. Bu değer gerçek bir canlılık analizi sonucu değil, kaynak kodda açıkça belirtildiği üzere bir yer tutucu değerdir (placeholder). Bu durum, kayıt döneminde sahte saldırılara karşı herhangi bir koruma yapılmadığı anlamına gelmektedir.

---

## 2. Biyometrik Doğruluk — FAR / FRR / EER / AUC

**Ne test edildi:** `POST /api/v1/verify` endpoint'i üzerinden 772 gerçek çift (aynı kişi, farklı fotoğraf) ve 4.828 sahte çift (farklı kişiler) skorlandı. ROC eğrisi, EER ve çalışma noktaları üretmek için eşik değeri taraması yapıldı.

**Sonuçlar:**

| Metrik | Değer |
|---|---|
| Gerçek (genuine) çift sayısı | 772 |
| Sahte (imposter) çift sayısı | 4.828 |
| AUC | 0,9943 |
| EER | %1,93 (eşik: 0,5885 mesafe) |
| Üretim eşiği (mesafe) | 0,45 |
| Üretim eşiğinde FAR | %0,27 |
| Üretim eşiğinde FRR | %4,40 |

**Yorum:**
- **AUC = 0,9943** — gerçek ve sahte çiftler arasında neredeyse mükemmel bir ayrım. Facenet512 modeli üst düzey bir performans sergilemektedir.
- **EER = %1,93** — LFW kıyaslama sonuçlarıyla rekabetçi bir değerdir (akademik en iyiler: <%1, iyi ticari sistemler: %0,5–2 arasında).
- **Mevcut üretim eşiğinde (0,45):**
  - FAR = %0,27 → yaklaşık 370 saldırıdan 1'i sistemi kandırabilir.
  - FRR = %4,40 → yaklaşık 23 meşru kullanıcıdan 1'i sisteme giremez.
- Mevcut eşik güvenliği ön planda tutuyor (düşük FAR) ancak belirgin bir yanlış ret oranı doğuruyor. Kullanım senaryosuna göre bu eşiğin yeniden ayarlanması değerlendirilebilir.

**Öneri:** Model kalitesi bir sorun değildir. Kullanıcı deneyimini iyileştirmek (FRR'yi düşürmek) bir öncelikse, FAR'ı %1'in altında tutacak şekilde üretim eşiğini 0,50–0,55 aralığına çekmeyi değerlendirin.

---

## 3. Çok Kiracılı İzolasyon — KRİTİK GÜVENLİK AÇIĞI

**Ne test edildi:** Örtüşen kullanıcı kimliklerine sahip iki ayrı kiracıya (ct-test-A ve ct-test-B) kayıt edilen 3 gerçek LFW kimliği kullanılarak 6 test vakası çalıştırıldı.

**Sonuçlar:**

| Vaka | Açıklama | Beklenti | Alınan | Karar |
|---|---|---|---|---|
| 01 | Aynı kişi, aynı kiracı | verified=True | verified=True | ✅ PASS |
| 02 | A kişisi, B kiracısında doğrulama yapıyor (B'de C kişisi var) | verified=False | verified=True | ❌ FAIL |
| 03 | C kişisi, A kiracısında doğrulama yapıyor (A'da A kişisi var) | verified=False | verified=True | ❌ FAIL |
| 04 | Var olan kiracıda bilinmeyen kullanıcı | 404 | 404 | ✅ PASS |
| 05 | Var olmayan kiracıda geçerli kişi | 404 | verified=True (200) | ❌ FAIL |
| 06 | Farklı kullanıcı, aynı kiracı | verified=False | verified=False | ✅ PASS |

**Kök Neden:**
`/verify` endpoint'i, embedding aramasında `tenant_id`'yi filtre olarak kullanmamaktadır. **Tüm kiracılara ait veriler üzerinde global arama** yapılmaktadır. Vaka 05 en kesin kanıttır: tamamen uydurulmuş bir kiracı adına yapılan istekte bile `verified=True` döndürülmekte ve mesafe değeri (0,212) meşru bir eşleşmeyle aynıdır.

**Etki:**
- A kiracısına kayıtlı herhangi bir kullanıcı, B kiracısında doğrulanabilir.
- Var olmayan bir kiracıya bile başarılı doğrulama yapılabilmektedir.
- Üniversite bölümleri veya farklı kurumlar gibi çok kiracılı üretim ortamlarında bu durum **kurumlar arası tam veri sızıntısı** anlamına gelmektedir.
- Önem Derecesi: **KRİTİK**

**Gerekli Düzeltme:**
Doğrulama use case'indeki embedding sorgusuna `WHERE tenant_id = :tenant_id AND user_id = :user_id` filtresi eklenmelidir. (tenant_id, user_id) kombinasyonu bulunamazsa 404 döndürülmelidir.

---

## 4. Canlılık ve Sahte Saldırı Tespiti

### 4A. MiniFASNet Canlılık Endpoint'i — KRİTİK

**Endpoint:** `POST /api/v1/liveness`

**Sonuç:** Her çağrıda HTTP 500 döndürüyor.

**Hata mesajı:** `Failed to initialize MiniFASNet: [Errno 13] Permission denied: '/nonexistent'`

**Kök Neden:** `MINIFAS_MODEL_PATH` ortam değişkeni `/nonexistent` olarak ayarlanmış — büyük olasılıkla docker-compose içinde gerçek model dosyası yoluyla hiç değiştirilmemiş bir yer tutucu.

**Etki:**
- Özel canlılık endpoint'i tamamen işlevsizdir.
- Bu hata giderilmeden BPCER ve APCER ölçümü yapılamaz.
- Kayıt sırasında döndürülen `liveness_score: 1.0` değeri, kaynak kodda açıkça belirtildiği üzere sabit kodlanmış bir yer tutucudur.
- Önem Derecesi: **KRİTİK**

**Gerekli Düzeltme:** Docker container içinde MiniFASNet model dosyasının doğru bağlama (mount) yoluna `MINIFAS_MODEL_PATH` ayarlanmalıdır.

---

### 4B. /verify Üzerinden Sentetik Sahte Saldırı — YÜKSEK RİSK

**Ne test edildi:** 60 kayıtlı kullanıcıya ait gerçek fotoğraflardan üretilen 4 farklı türde sentetik sahte görüntü, her kullanıcının kendi kimliğine karşı `/verify`'a gönderildi.

**Saldırı türleri ve sonuçları:**

| Saldırı Türü | Açıklama | Kabul Edilen | Toplam | APCER |
|---|---|---|---|---|
| PRINT_MILD | Hafif bulanıklaştırma + JPEG q=70 | 54 | 58 | **%93,1** |
| SCREEN_REPLAY | Tarama çizgileri + sıcak renk kayması | 54 | 57 | **%94,7** |
| BORDER_PRINT | Beyaz kenarlıklı baskı fotoğrafı | 54 | 56 | **%96,4** |
| PRINT_STRONG | Yoğun bulanık + vignette + q=45 | 39 | 57 | **%68,4** |
| **Ortalama APCER** | | | | **%88,2** |
| GENUINE (BPCER) | Kayıtlı kişinin gerçek fotoğrafı | 52 | 55 | %5,5 yanlış red |

**Yorum:**
- `/verify` endpoint'inin sahte saldırılara karşı herhangi bir koruması bulunmamaktadır — yalnızca yüz embedding'lerini karşılaştırmaktadır.
- Basılı bir fotoğraf, telefon ekranından tekrar oynatılan görüntü ya da kayıt fotoğrafının hafifçe bozulmuş bir kopyası ortalamada %88 oranında gerçekmiş gibi kabul edilmektedir.
- Bu, herhangi bir kayıtlı kullanıcının fiziksel fotoğrafı kullanılarak sisteme erişilebileceği anlamına gelmektedir.

**Gerekli Düzeltme:** Yukarıdaki BUG-02 düzeltildikten sonra, canlılık kontrolü her `/verify` çağrısına entegre edilmelidir.

---

### 4C. Puzzle Canlılık Meydan Okuma-Yanıt Sistemi — KRİTİK BYPASS

**Endpoint:** `POST /api/v1/liveness/verify` (önce `/liveness/generate-puzzle` ile puzzle oluşturularak)

**Ne test edildi:** Göz kırpma, gülümseme, kafa çevirme gibi eylem adımlarından oluşan challenge-response canlılık sistemine karşı 10 saldırı senaryosu.

**Sonuçlar:**

| Test | Beklenti | Sonuç | Karar |
|---|---|---|---|
| T1 — Yanlış eylem adları | reddedildi | reddedildi | ✅ PASS |
| **T2 — Doğru eylemler, sahte güven skoru** | **reddedildi** | **liveness_confirmed=True** | ❌ **KRİTİK** |
| T3 — Aynı puzzle_id'nin tekrar kullanımı | reddedildi | PUZZLE_ALREADY_COMPLETED | ✅ PASS |
| T4 — Var olmayan puzzle_id | reddedildi | PUZZLE_NOT_FOUND | ✅ PASS |
| T5 — Yanlış adım sırası | reddedildi | reddedildi | ✅ PASS |
| T6 — confidence=0,3 (0,6 minimumun altı) | reddedildi | reddedildi | ✅ PASS |
| T7 — Adım süresi 0,1s (0,5s minimumun altı) | reddedildi | reddedildi | ✅ PASS |
| T8 — 10 dakika önceki zaman damgası | reddedildi | reddedildi | ✅ PASS |
| T9 — spot_frames olarak statik fotoğraf | reddedildi | liveness_confirmed=True | ⚠️ UYARI |
| T10 — Zorluk seviyesi adım sayıları | doğru sayılar | doğru sayılar | ✅ PASS |

**Kritik Bulgu (T2):**
Puzzle sistemi yalnızca iki API çağrısıyla tamamen atlatılabilmektedir:

```
Adım 1: POST /api/v1/liveness/generate-puzzle
        Cevap: puzzle_id + adım adları (örn. ["blink", "smile", "turn_left"])

Adım 2: POST /api/v1/liveness/verify
        Gövde: {
          "puzzle_id": "<1. adımdan gelen id>",
          "results": [
            {"action": "blink",      "start_timestamp": <şimdi>,    "end_timestamp": <şimdi+1.5>, "confidence": 0.99},
            {"action": "smile",      "start_timestamp": <şimdi+1.7>,"end_timestamp": <şimdi+3.2>, "confidence": 0.99},
            {"action": "turn_left",  "start_timestamp": <şimdi+3.4>,"end_timestamp": <şimdi+4.9>, "confidence": 0.99}
          ]
        }
        Cevap: liveness_confirmed=True, overall_score=99.0
```

Kamera gerekmez, gerçek eylem gerekmez. Saldırganın yalnızca generate-puzzle'dan gelen adım adlarını bilmesi ve geçerli zaman damgalarıyla sahte güven değerleri göndermesi yeterlidir.

**Kök Neden:**
`confidence` alanı istemci tarafından gönderilmekte ve sunucu tarafında doğrulanmadan güvenilmektedir. Gerçek kamera görüntülerini analiz edebilecek sunucu taraflı `spot_frames` kontrolü kodda mevcuttur ancak MiniFASNet dedektörü başlatılamadığı için (BUG-02) devre dışı kalmaktadır.

**Gerekli Düzeltmeler:**
1. MiniFASNet model yolunu düzelt (BUG-02) — sunucu taraflı spot kontrolünü etkinleştir.
2. `spot_frames` alanını zorunlu hale getir — bu alan olmadan gelen tüm verify isteklerini reddet.
3. Spot frame'leri sunucu tarafında analiz et; canlılık skoru eşiğin altındaysa reddet.

---

## 5. Edge Case (Sınır Durum) ve Girdi Doğrulama

**Ne test edildi:** `/enroll` ve `/verify`'a geçersiz formatlar, aşırı boyutlar, boş görüntüler ve kimlik doğrulama atlatma girişimleri içeren 10 sınır durum girdisi gönderildi.

**Sonuçlar:**

| Vaka | Girdi | Beklenti | Alınan | Karar |
|---|---|---|---|---|
| 01 | Tamamen beyaz görüntü | 400 | 400 | ✅ PASS |
| 02 | 10×10 piksel görüntü | 400 | 400 | ✅ PASS |
| 03 | Düz beyaz 640×480 | 400 | 400 | ✅ PASS |
| 04 | Tamamen siyah 640×480 | 400 | 400 | ✅ PASS |
| 05 | 12 MB dosya | 413 | 413 | ✅ PASS |
| **06** | **GIF dosyası** | **400/415** | **500** | ❌ **FAIL** |
| 07 | Bozuk JPEG | 400 | 400 | ✅ PASS |
| 08 | JPEG görünümlü metin dosyası | 400 | 400 | ✅ PASS |
| 09 | Enroll ve verify'da aynı görüntü | verified=True | verified=True | ✅ PASS |
| **10** | **Geçersiz API anahtarı** | **401** | **200** | ❌ **FAIL** |

**Bug 05 — GIF Formatı 500 Döndürüyor:**
GIF dosyası gönderildiğinde sunucu, düzgün bir 400/415 yerine HTTP 500 İç Sunucu Hatası döndürmektedir. Format doğrulaması API sınırında değil, model pipeline'ının içinde gerçekleştiği için GIF sisteme girmekte ve model içinde çökmektedir.

**Bug 06 — API Kimlik Doğrulama Uygulanmıyor:**
`X-Api-Key: INVALID_KEY_XYZ_000` içeren bir istek HTTP 200 ile kabul edilmektedir. API anahtarı doğrulaması ya uygulanmamıştır ya da mevcut dağıtımda devre dışıdır. Endpoint adresini bilen herhangi bir kişi kimlik bilgisi olmadan sisteme erişebilmektedir.

---

## 6. Performans

**Ne test edildi:** Sıralı gecikme (30 çağrı), eşzamanlı yük (20 paralel istek), hız limitleme (40 ardışık çağrı) ve arama gecikmesi (5 çağrı).

### P1 — Tek İstek Gecikmesi

| Metrik | Değer | Hedef | Durum |
|---|---|---|---|
| Min | 0,201s | — | — |
| p50 | 0,227s | — | — |
| p95 | 0,423s | ≤ 1,5s | ✅ PASS |
| p99 | 0,424s | — | — |
| Max | 0,424s | — | — |

Tek istek performansı hedefin çok altındadır. İlk 1–2 istek ~0,4s sürmekte (model ısınma), ardından ~0,22s'ye düşmektedir.

### P2 — Eşzamanlı Yük (20 Paralel İstek)

| Metrik | Değer |
|---|---|
| Toplam süre (20 istek) | 6,95s |
| İstek başına p50 | 4,257s |
| İstek başına p95 | 5,689s |
| Hata sayısı | 0 / 20 |

Eşzamanlı yük altında çökme veya hata oluşmamıştır — olumlu. Ancak istek başına gecikme, sıralı kullanımdaki 0,22s'den 4–5s'ye yükselmiştir (yaklaşık 17–25 kat yavaşlama). Bu, DeepFace modelinin eşzamanlı HTTP isteklerine rağmen sıralı olarak çalıştırılmasından kaynaklanmaktadır.

**Not:** Biyometrik API sunucusu şu anda **%94 bellek kullanımındadır**. Sürekli eşzamanlı yük altında OOM (Bellek Yetersizliği) hataları oluşabilir.

### P3 — Hız Limitleme

40 ardışık çağrıda hız limiti tetiklenmedi. Test ortamında `RATE_LIMIT_ENABLED` ortam değişkeni `False` olarak ayarlıdır. Üretim yapılandırmasında `True` olduğu doğrulanmalıdır. Hız limiti olmadan sistem kaba kuvvet saldırılarına açık kalmaktadır.

### P4 — Arama Gecikmesi

| Metrik | Değer |
|---|---|
| Medyan | 0,256s |
| Maksimum | 0,305s |

553 embedding üzerinde arama şu an hızlıdır. Ancak pgvector **HNSW index'i henüz yapılandırılmamıştır** (TODO olarak işaretli). Index olmadan veritabanı sıralı tarama yapmaktadır. Kullanıcı sayısı büyüdükçe beklenen gecikme:

| Kullanıcı | Tahmini gecikme (index yok) |
|---|---|
| 553 (mevcut) | ~0,25s |
| 10.000 | ~4–5s |
| 100.000 | ~40–50s |

---

## 7. Tam Hata Kaydı

| ID | Önem | Alan | Açıklama | Gerekli Aksiyon |
|---|---|---|---|---|
| BUG-01 | 🔴 KRİTİK | Çok Kiracılı | `/verify`'da `tenant_id` uygulanmıyor — global embedding araması yapılıyor | DB sorgusuna `WHERE tenant_id = :tenant_id` ekle |
| BUG-02 | 🔴 KRİTİK | Canlılık | MiniFASNet model yolu `/nonexistent` olarak ayarlı — endpoint 500 döndürüyor | docker-compose'da doğru `MINIFAS_MODEL_PATH` ayarla |
| BUG-03 | 🔴 KRİTİK | Canlılık | Puzzle canlılık bypass: sahte güven + doğru eylem adları → `liveness_confirmed=True` | `spot_frames`'i zorunlu yap; BUG-02'yi düzelt |
| BUG-04 | 🟠 YÜKSEK | Sahte Saldırı | `/verify`'ın sahte saldırı direnci yok — ortalama APCER %88,2 | BUG-02 düzeltmesinden sonra canlılık kontrolünü verify akışına entegre et |
| BUG-05 | 🟡 ORTA | Girdi Doğrulama | GIF girdisi 400/415 yerine 500 döndürüyor | Model pipeline'ından önce format doğrulama ekle |
| BUG-06 | 🟡 ORTA | Kimlik Doğrulama | Geçersiz API anahtarı 200 döndürüyor | Tüm endpoint'lerde API anahtarı doğrulamasını uygula |
| BUG-07 | 🟡 ORTA | Performans | 20 eşzamanlı kullanıcıda 17–25× gecikme artışı | Asenkron model worker'ları veya istek kuyruğu yapılandır |
| BUG-08 | 🟡 ORTA | Performans | pgvector HNSW index eksik — ölçeklenince arama bozulacak | `face_embeddings` tablosuna HNSW index ekle |
| BUG-09 | 🟡 ORTA | Performans | Hız limitleme devre dışı (`RATE_LIMIT_ENABLED=False`) | Üretim yapılandırmasında hız limitlemenin etkin olduğunu doğrula |
| BUG-10 | 🟢 DÜŞÜK | Kayıt | Kayıt sırasında `liveness_score: 1.0` sabit kodlanmış yer tutucu | TODO olarak kaydet — çağıranları yanıltmaktadır |

---

## 8. Düzgün Çalışan Özellikler

- **Yüz tanıma modeli kalitesi** — AUC 0,9943 mükemmeldir. Facenet512 doğru entegre edilmiş ve güvenilir embedding'ler üretmektedir.
- **Eşik kalibrasyonu** — 0,45'lik üretim eşiği, FAR=%0,27 ile makul bir çalışma noktası sunmaktadır.
- **Girdi doğrulama (çoğu durum)** — Boş görüntüler, küçük görüntüler, siyah görüntüler, bozuk JPEG'ler, aşırı büyük dosyalar ve metin dosyaları doğru şekilde reddedilmektedir.
- **Puzzle yapısal bütünlüğü** — Tekrar saldırıları, süresi dolmuş puzzle'lar, yanlış eylem sıraları, düşük güven değerleri ve geçmiş zaman damgaları puzzle canlılık sistemi tarafından doğru şekilde tespit edilip reddedilmektedir.
- **Aynı kiracı içinde kullanıcı izolasyonu** — A kiracısındaki bir kullanıcı, aynı kiracıdaki farklı bir kullanıcı olarak doğrulanamaz.
- **İstemci tarafı bypass alanları görmezden geliniyor** — Verify isteğine `liveness_passed=true` gibi fazladan alanlar eklenmesinin herhangi bir etkisi yoktur; sunucu bu alanları doğru şekilde yok saymaktadır.
- **Dosya boyutu limiti** — 12 MB dosyalar HTTP 413 ile doğru şekilde reddedilmektedir.

---

## 9. Öncelikli Aksiyon Planı

| Öncelik | Aksiyon | Sorumlu | Engel |
|---|---|---|---|
| P0 — Acil | docker-compose'da `MINIFAS_MODEL_PATH` düzelt (BUG-02) | DevOps | BUG-03, BUG-04 |
| P0 — Acil | Verify sorgusuna `WHERE tenant_id` filtresi ekle (BUG-01) | Backend | Üretim lansmanı |
| P0 — Acil | Puzzle verify'da `spot_frames`'i zorunlu yap ve eksik olursa reddet (BUG-03) | Backend | Üretim lansmanı |
| P1 — Lansmandan önce | Tüm endpoint'lerde API anahtarı doğrulamasını uygula (BUG-06) | Backend | Güvenlik uyumluluğu |
| P1 — Lansmandan önce | Model pipeline'ından önce GIF/BMP vb. için format doğrulama ekle (BUG-05) | Backend | |
| P1 — Lansmandan önce | Üretim yapılandırmasında `RATE_LIMIT_ENABLED=True` olduğunu doğrula (BUG-09) | DevOps | |
| P2 — Kısa vadeli | Eşzamanlı gecikmeyi azaltmak için asenkron worker'lar yapılandır (BUG-07) | Backend/DevOps | |
| P2 — Kısa vadeli | `face_embeddings` tablosuna pgvector HNSW index ekle (BUG-08) | Backend/DB | |
| P3 — Devam eden | Gerçek sahte saldırı veri seti (NUAA/Replay-Attack) indir ve APCER ölç | QA | |
| P3 — Devam eden | Sunucu belleğini izle (%94'te) ve OOM uyarısı kur | DevOps | |

---

*Rapor, `C:\Users\hp\fivucsas-test\` dizinindeki otomatik test paketinden oluşturulmuştur.*
*Test script'leri yeniden çalıştırılabilir — düzeltmeler uygulandıktan sonra her adım ayrı ayrı çalıştırılabilir.*
