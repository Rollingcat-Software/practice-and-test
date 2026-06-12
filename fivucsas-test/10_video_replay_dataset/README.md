# 10 — Video Replay Dataset PAD Evaluation (Otomatik)

## Amaç

Bizim 04 testimiz tek-kare/sentetik görüntülerle, 09 ise manuel webcam ile spoof
testi yapıyordu. Bu modül **gerçek bir video PAD dataset'i** kullanarak, videoları
doğrudan amispoof'un arkasındaki **session engine'e** besleyip ISO 30107-3 metriklerini
(APCER / BPCER / ACER) **otomatik** hesaplar. Webcam'e ekran doğrultmaya gerek yok.

## Neden geçerli?

PAD dataset'lerindeki **attack** videoları zaten ekran/baskının kameraya gösterilip
kaydedilmiş hali — moiré, bezel, çift-yakalama artefaktları videoya gömülü. Dolayısıyla
attack videosunu dedektöre doğrudan beslemek = replay saldırısını gerçekleştirmek.
Akademik PAD değerlendirmesi tam da böyle yapılır.

## Hedef: ~200 video yeterli

Proje için **100 genuine + 100 attack = 200 video** fazlasıyla yeterli (APCER/BPCER'ı
~%1 çözünürlükle ölçer). Bu yüzden devasa dataset indirmeye gerek yok — küçük bir
dataset seç veya büyük olanı indirip script ile 200'e düşür (`--max-per-class 100`).

### Seçenek A — Küçük dataset (önerilen, az indirme)

- **CASIA-FASD** (~600 video, .avi, birkaç GB) — Kaggle'da "casia fasd" diye ara,
  birkaç mirror var. 200'e subsample ederiz.
- **MSU-MFSD** (~280 video) — neredeyse tam 200 hedefinde; akademik istek gerekir.

### Seçenek B — Büyük dataset (kolay erişim ama büyük indirme)

**Kaggle — "Antispoofing Replay Dataset" (26.000+ video, onlarca GB)**
`https://www.kaggle.com/datasets/trainingdatapro/antispoofing-replay-dataset`
Sadece 200 kullanacaksan indirmesi israf; ama erişimi en kolay olan bu.

```powershell
pip install kaggle
# Kaggle > Settings > API > Create New Token → kaggle.json'u %USERPROFILE%\.kaggle\ altına koy
kaggle datasets download -d <dataset-slug> -p C:\Users\hp\Documents\GitHub\Dataset\pad-videos --unzip
```

> Hangi dataset olursa olsun script çalışır — etiketi klasör/dosya adındaki
> anahtar kelimeden çıkarır (genuine/real/live ↔ attack/spoof/replay/print).
> İndirme bitince önce `--dry-run` ile etiket dağılımını doğrula.

### Seçenek C — YouTube injection attack (otomatik, dataset indirmeden)

Hazır PAD dataset'i yerine YouTube yüz videolarını **video-injection saldırısı**
olarak kullan. Ham videoyu doğrudan dedektöre besleriz → sanal-kamera/DeepFaceLive
tarzı enjeksiyonu simüle eder. Bu **ekran-replay değil**; sistemin en zor saldırı
türüne (doğrudan injection) karşı zafiyetini ölçer.

```powershell
pip install yt-dlp          # + ffmpeg PATH'te olmalı
```

**Önerilen: sorgu listesi (çakışmasız, güncel videolar)** — `youtube_queries.txt`
hazır küratörlü sorgularla geliyor (no-copyright, yüz-merkezli):
```powershell
python 10_video_replay_dataset\fetch_youtube.py --queries-file 10_video_replay_dataset\youtube_queries.txt --per-query 4 --seconds 15
```
8 sorgu × 4 = ~32 aday; dedup + yüz-tespiti sonrası ~20 kullanılabilir video iner.

**Alternatif: elle URL listesi** — `youtube_urls.txt`'e URL yapıştır:
```powershell
python 10_video_replay_dataset\fetch_youtube.py --urls 10_video_replay_dataset\youtube_urls.txt --seconds 15
```

Sonra (her iki yolda da):
```powershell
# indirilenler youtube_injection/ altına injection_NNN.mp4 olarak iner
python 10_video_replay_dataset\batch_video_eval.py --dataset 10_video_replay_dataset\youtube_injection --attack-keyword injection
```

- Bu yol **genuine sınıfı gerektirmez** → sadece **APCER-injection** raporlanır
  (enjekte videonun "LIVE" sanılma oranı). Beklenen: yüksek → injection zafiyeti kanıtı.
- ToS notu: yt-dlp ile public içerik araştırma amaçlı indirilir; raporda belirt.

## Ortam Kurulumu (spoof-detector bağımlılıkları)

Script, spoof-detector'ın pipeline'ını import eder. O paketin bağımlılıkları kurulu olmalı:

```powershell
cd C:\Users\hp\Documents\GitHub\Rollingcat-Software\FIVUCSAS\practice-and-test\spoof-detector
pip install -r requirements.txt
```

⚠️ **Python sürümü**: MediaPipe genelde Python 3.11/3.12 için wheel sunar; **3.13'te
çalışmayabilir**. Senin sistemin 3.13.7. MediaPipe kurulmazsa, 3.12'lik ayrı bir venv aç:
```powershell
py -3.12 -m venv C:\Users\hp\spoof-venv
C:\Users\hp\spoof-venv\Scripts\activate
pip install -r requirements.txt
```
Script'i bu venv ile çalıştır.

## Çalıştırma

Önce etiket dağılımını kontrol et (inference yapmadan, hızlı):

```powershell
python 10_video_replay_dataset\batch_video_eval.py `
  --dataset C:\Users\hp\Documents\GitHub\Dataset\antispoof-replay `
  --dry-run
```

Çıktı: kaç genuine / kaç attack bulundu + örnek yollar. Etiketleme yanlışsa
`--genuine-keyword` / `--attack-keyword` ile düzelt.

Sonra gerçek değerlendirme — **hedef 200 video**:

```powershell
python 10_video_replay_dataset\batch_video_eval.py `
  --dataset C:\Users\hp\Documents\GitHub\Dataset\pad-videos `
  --max-per-class 100 `
  --frame-stride 2 `
  --max-frames 300
```

- `--max-per-class 100` → 100 genuine + 100 attack = **200 video** (proje için yeterli)
- `--frame-stride 2`    → her 2 kareden 1'ini işle (hız)
- `--max-frames 300`    → video başına en fazla 300 kare (~10s @30fps)

İlk denemede daha da hızlı görmek istersen `--max-per-class 25` ile başla (~50 video, ~birkaç dk).

## Çıktılar

| Dosya | İçerik |
|---|---|
| `video_results.csv` | Video başına: ground_truth, verdict, confidence, dominant_threat, correct |
| `summary.txt` | APCER / BPCER / ACER + ISO Grade |

## Yorum

- **APCER** burada = replay/print videolarının "LIVE" sanılma oranı. 04'teki sentetik
  testte sistemin en zayıf noktasıydı (~%30). Gerçek dataset bunu doğrular/çürütür.
- **BPCER** = gerçek videoların "SPOOF" sanılma oranı (kullanıcı deneyimi).
- Bu modül **biometric-processor Docker'dan bağımsız** (spoof-detector ayrı paket) —
  arka planda koşan 8-test'i etkilemez, ama ikisi de CPU/ML kullandığı için aynı anda
  koşturmamak daha hızlı olur.

## Not

Script, her video için pipeline'ı yeniden kurar (analizör durumu videolar arası
sızmasın diye). MediaPipe/MiniFASNet oturumları process-genelinde paylaşımlı olduğu
için bu maliyet kabul edilebilir. Yine de yüzlerce video için `--frame-stride` ve
`--max-frames` ile süreyi kontrol et.
