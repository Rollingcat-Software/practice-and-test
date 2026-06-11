# Gesture Analyser — Web (El + Yüz Canlılık Görevleri)

`Gesture Analyser` motorunu (el takibi) **hiç değiştirmeden** tarayıcıdan
çalıştıran ve üstüne **yüz hareketi** görevleri ekleyen React + FastAPI köprü
uygulaması. **23 gerçek-zamanlı canlılık görevi**: 9 el + 14 yüz.

> Çalıştırma adımları için **[SETUP.md](SETUP.md)** dosyasına bakın.

## Klasör yapısı

```
gesture analyser/
├── backend/            FastAPI + WebSocket köprüsü
│   ├── bridge.py            # motoru (Gesture Analyser) bulup sys.path'e ekler
│   ├── server.py           # /ws WebSocket endpoint + frame işleme
│   ├── state_serializer.py # oturum durumunu JSON'a çevirir
│   ├── face_tracker.py     # MediaPipe FaceLandmarker (478 nokta + 52 blendshape)
│   ├── face_session.py     # 14 yüz görevi dedektörü
│   ├── shape_bridge.py     # robust şekil-çizim oturumu (DTW)
│   ├── face_landmarker.task
│   └── requirements.txt
├── frontend/           Vite + React + TypeScript
│   └── src/                # UI, kartlar, WS hook, kamera + canvas
└── Gesture Analyser/   El takip motoru (gömülü — hand_landmarker.task dahil)
```

`Gesture Analyser` motorunun **hiçbir kaynak dosyası değiştirilmez**;
`backend/bridge.py` onu proje kökünde otomatik bulur (eski adı `PythonProject1`
idi, bridge her iki adı da tanır).

## Mimari

```
┌────────────────────────────┐  jpeg frame (base64, WS)
│ Tarayıcı (React)           │ ──────────────────────────►  ┌──────────────────────┐
│  • getUserMedia (webcam)   │                              │ FastAPI köprüsü      │
│  • El/Yüz görev kartları   │  durum + landmark (JSON)     │  ├ HandTracker (motor)│
│  • Canvas HUD + overlay    │ ◄──────────────────────────  │  └ FaceTracker (yeni) │
└────────────────────────────┘                              └──────────────────────┘
```

İstek-yanıt akış kontrolü: tarayıcı bir frame yollar, backend yanıtı gelmeden
yenisini yollamaz → kuyruk birikmez, gerçek-zamanlı kalır.

## Görevler

### El Hareketi Görevleri (9) — `GameManager` modları
| Kart | Mod | Modül |
|------|-----|-------|
| Parmak Say | 0 | `gesture_session` |
| El Salla | 2 | `liveness_session` (WAVE) |
| Avuç Çevir | 2 | `liveness_session` (HAND_FLIP) |
| Parmak Tıklama | 2 | `liveness_session` (FINGER_TAP) |
| Sıkıştır | 4 | `finger_touch_session` |
| Ce-e | 2 | `liveness_session` (PEEK_A_BOO) |
| Şekil Çiz | 5 | bridge şekil oturumu (DTW) |
| Sıralı Görev | 3 | `sequential_session` |
| Matematik | 1 | `math_session` |

Liveness kartları, ilgili komuta odaklanır (backend `liveness.reset()` ile).

### Yüz Görevleri (14) — bridge `FaceTaskSession`
Göz Kırp · Sol/Sağ Gözü Kapat · Geniş Gülümse · Ağzı Aç · Başı Sola/Sağa Çevir ·
Yukarı/Aşağı Bak · Her İki/Sol/Sağ Kaşı Kaldır · Başı Salla (Evet/Hayır)

MediaPipe FaceLandmarker blendshape'leri + landmark-tabanlı baş pozu (yaw/pitch/
roll) kullanır. Göz/kaş/poz görevleri **baseline-göreli** algılar (nötr ölçülür,
üstüne çıkış değerlendirilir). Baş salla için osilasyon filtresi.

## WebSocket protokolü (özet)

Client → Server:
```json
{ "type": "frame", "data": "data:image/jpeg;base64,..." }
{ "type": "set_mode", "mode": 0, "prefer_cmd": "WAVE" }
{ "type": "set_face_task", "task": "blink" }
{ "type": "restart" }
```

Server → Client: her frame için `{"type":"state", ...}` — el modlarında `hands`
+ moda özgü `session`; yüz modunda `face_groups` (mesh) + `session` (durum,
ilerleme, ham blendshape değerleri).

## Notlar
- Her WS bağlantısı kendi tracker + oturum örneğine sahip (izolasyon).
- Frame JPEG q=0.92, hedef ~30 fps (`TARGET_FPS`, `TaskRunner.tsx`).
- Aynı anda tek tracker çalışır: el modlarında HandTracker, yüz modunda
  FaceTracker (mod değişiminde otomatik geçiş).
- HUD/mesh çizimi tamamen frontend tarafında (canvas).
