# Kurulum ve Çalıştırma Rehberi

El (9) ve Yüz (14) hareketi canlılık görevlerini tarayıcıda çalıştıran proje.
MediaPipe HandLandmarker + FaceLandmarker kullanır.

## Klasör yapısı (kendi kendine yeten)

```
Gesture B-F/
├── backend/            FastAPI + WebSocket sunucusu (+ face_landmarker.task)
├── frontend/           React + Vite arayüzü
├── Gesture Analyser/   El takip motoru (gömülü — hand_landmarker.task dahil)
└── SETUP.md            (bu dosya)
```

`Gesture Analyser` motoru projenin köküne gömülüdür; `backend/bridge.py` onu
otomatik bulur — **yol ayarı gerekmez**. (Bu motorun eski adı `PythonProject1`
idi; bridge her iki adı da tanır.)

---

## Gereksinimler

| Araç | Sürüm |
|------|-------|
| **Python** | 3.10 veya üzeri (3.13'te test edildi) |
| **Node.js** | 18 veya üzeri (npm ile birlikte) |
| **Webcam** | Gerekli |
| **Tarayıcı** | Güncel Chrome / Edge / Brave / Firefox |

İndirme: Python → https://python.org · Node.js → https://nodejs.org

---

## ⚠ ZIP'lemeden önce (gönderen kişi)

Şu klasörler **taşınabilir değildir**, ZIP'e koymayın (alıcı yeniden kuracak):

```
backend/.venv
frontend/node_modules
**/__pycache__
frontend/dist
```

> `hand_landmarker.task` (Gesture Analyser içinde) ve `face_landmarker.task`
> (backend içinde) **mutlaka ZIP'te olmalı** — bunlar model dosyaları.

---

## Kurulum (alıcı — bir kez)

İki ayrı terminal kullanın. Windows PowerShell örneği:

### 1) Backend

```powershell
cd "<proje yolu>\Gesture B-F\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> macOS / Linux: `python3 -m venv .venv` → `source .venv/bin/activate` → `pip install -r requirements.txt`

İlk `pip install` ~500 MB indirir (mediapipe, opencv). Tek seferlik.

### 2) Frontend

```powershell
cd "<proje yolu>\Gesture B-F\frontend"
npm install
```

---

## Çalıştırma (her seferinde)

**Terminal 1 — Backend**
```powershell
cd "<proje yolu>\Gesture B-F\backend"
.\.venv\Scripts\Activate.ps1
python server.py
```
Beklenen: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 — Frontend**
```powershell
cd "<proje yolu>\Gesture B-F\frontend"
npm run dev
```
Beklenen: `Local: http://localhost:5173/`

Tarayıcıda **http://localhost:5173** açın → kameraya izin verin → bir görev kartına girin.

Durdurmak için her terminalde **Ctrl+C**.

---

## Sorun giderme

- **"El takip motoru bulunamadı"** → `Gesture Analyser` klasörünün projenin
  kökünde (backend/ ve frontend/ yanında) olduğundan emin olun. Ya da
  `python server.py` öncesi `$env:GESTURE_ANALYSER_PATH = "tam\yol"`.
- **`pip install` mediapipe hatası** → Python sürümünüz 3.10–3.13 arası mı
  kontrol edin (mediapipe çok yeni/eski sürümlerde olmayabilir).
- **Tarayıcı "localhost'a bağlanılamadı"** → Backend ve frontend'in ikisinin
  de açık olduğundan emin olun. Frontend `/ws` ve `/api`'yi backend'e
  (127.0.0.1:8000) proxy'ler.
- **Kamera açılmıyor** → Tarayıcı kamera iznini kontrol edin; `localhost`
  üzerinden eriştiğinizden emin olun (tarayıcılar başka kaynaklarda kamerayı
  engeller).
- **Port dolu** → 8000 veya 5173 başka bir uygulamada açıksa kapatın.
