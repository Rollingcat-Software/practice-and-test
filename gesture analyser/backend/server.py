"""
FastAPI Gesture Analyzer Sunucusu
=================================
Tarayıcıdan webcam frame'lerini WebSocket üzerinden alır, PythonProject1'in
HandTracker + GameManager modüllerine besler ve durum + landmark'ları geri
döndürür.

PythonProject1 kaynak dosyaları HİÇ değiştirilmez — bridge.py ile import edilir.

WebSocket protokolü
-------------------
Client → Server:
  {"type": "frame", "data": "<base64 jpeg>"}
  {"type": "set_mode", "mode": <0..5>}
  {"type": "restart"}
  {"type": "cycle_eval"}            # sadece mode 5 için

Server → Client:
  {"type": "state", "mode": ..., "mode_name": ..., "hands": [...], "session": {...}, "fps": ...}
  {"type": "error", "message": "..."}
  {"type": "ready", "modes": [...]}
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bridge import HandTracker, GameManager, MODE_NAMES  # PythonProject1
from state_serializer import serialize_state, serialize_bridge_shape, serialize_face
from shape_bridge import BridgeShapeSession
# Sequential session içindeki StepResult — çizim adımlarını atlarken kullanılıyor.
from sequential_session import StepResult  # noqa: E402
# Yüz görevleri (PythonProject1'de yok — bridge'e özgü yeni katman).
from face_tracker import FaceTracker  # noqa: E402
from face_session import FaceTaskSession, VALID_FACE_TASKS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gesture-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Sunucu başlatıldı.")
    yield
    log.info("Sunucu kapatılıyor.")


app = FastAPI(title="Gesture Analyzer Web Bridge", lifespan=lifespan)

# Geliştirme sırasında frontend farklı portta çalışacak.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/modes")
def list_modes():
    return {"modes": [{"index": i, "name": name} for i, name in enumerate(MODE_NAMES)]}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/face-tasks")
def list_face_tasks():
    return {"tasks": sorted(VALID_FACE_TASKS)}


def _decode_b64_jpeg(data_b64: str) -> np.ndarray | None:
    """data:image/jpeg;base64,... önekini de tolere eder."""
    if not data_b64:
        return None
    if "," in data_b64:
        data_b64 = data_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_b64, validate=False)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _bias_liveness_command(gm, target_name: str, max_tries: int = 500) -> bool:
    """Liveness'ın AKTİF komutunu istenen tipe getirir.

    PythonProject1'e HİÇ dokunmadan, yalnızca public ``reset()`` API'sini ve
    ``current_cmd`` okumasını kullanır. LivenessChallenge her reset()'te 5
    komutluk kuyruğu rastgele kurduğu için, current_cmd.cmd_type istenen tip
    olana kadar reset() çağırırız. Bu sayede her liveness kartı (El Salla →
    WAVE, Avuç Çevir → HAND_FLIP, ...) adıyla uyumlu komutu gösterir.

    Başarılıysa True döner; max_tries içinde bulunamazsa False (mevcut rastgele
    komutla devam edilir).
    """
    if not target_name:
        return False
    live = gm.liveness
    for _ in range(max_tries):
        cmd = getattr(live, "current_cmd", None)
        if cmd is not None and cmd.cmd_type.name == target_name:
            return True
        live.reset()
    return False


# Mod-bazlı tracker ayarları:
#   • Şekil çizimi (mod 5): uzun ve sürekli kayıt; el kaybını uzun köprüler
#     (max_lost=8 ≈ 0.27s) ve düşük güvenle (0.35) el-takibi bırakmaz.
#   • Diğer modlar (liveness/gesture/math/touch): hızlı el hareketleri var,
#     kararsız tracker "geri kalma" hissi yaratıyor. Daha duyarlı ayar (kısa
#     hold + standart güven) doğal his verir; anlık glitch'ler için max_lost=3
#     yine config.yaml=2 varsayılanından biraz cömert.
def _tracker_params(mode_idx: int) -> tuple[int, int, float]:
    """(num_hands, max_lost_frames, min_tracking_confidence)"""
    if mode_idx == 5:
        return (1, 8, 0.35)   # şekil çizimi: tek el, uzun tolerans
    return (2, 3, 0.5)        # diğerleri: iki el, hızlı tepki


def _make_tracker_for(mode_idx: int) -> HandTracker:
    num_hands, max_lost, tracking_conf = _tracker_params(mode_idx)
    return HandTracker(
        max_num_hands=num_hands,
        max_lost_frames=max_lost,
        min_tracking_confidence=tracking_conf,
    )


# Sıralı görevde atlanacak adım tipleri. Şekil Çiz başlı başına bir kart olduğu
# ve sequential'da şablon görüntülenmediği için DRAW_CIRCLE/SQUARE adımları
# kullanıcı için belirsiz kalıyor. Bridge'de bu adımları otomatik PASSED işaretleyip
# atlıyoruz (PythonProject1'e dokunmadan, dataclass field'larına yazarak).
_SKIPPED_SEQ_TYPES = {"draw_circle", "draw_square"}


def _skip_drawing_seq_steps(seq) -> None:
    """Sıralı görev, çizim tipli adıma geldiyse onu atla. Birden fazla çizim
    adımı arka arkayaysa hepsini atlar.
    """
    while (seq.current_step is not None
           and seq.current_step.step_type in _SKIPPED_SEQ_TYPES):
        seq.step_results[seq.current_step_idx] = StepResult.PASSED
        # _advance() davranışını manuel taklit — private metoda dokunmuyoruz.
        seq.current_step_idx += 1
        seq._hold_start = None
        seq._step_start = None
        seq._step_done_at = None
        seq._baseline_depth = None
        seq._current_depth = None
        seq._flip_baseline_z = None
        seq._peekaboo_phase = 0
        seq._peekaboo_hidden_at = None
        seq._wave.reset()
        seq._shape.reset()
        seq._was_drawing = False
        seq._validator.clear_buffers()
        # State güncelle (son adıma geçtiysek COMPLETE)
        from sequential_session import SeqState
        if seq.current_step_idx >= len(seq.steps):
            seq.state = SeqState.COMPLETE
        else:
            seq.state = SeqState.ACTIVE


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WS bağlantı kabul edildi: %s", ws.client)

    # Mode 5 (şekil çizimi) için robust, süreye-kadar-çiz oturumu. GameManager'ın
    # ShapeTracerSession'ı kırılgan tetikler (tek-frame yumruk/el-kaybı) içerdiği
    # için mode 5'te onu DEĞİL bunu kullanıyoruz (PythonProject1 matematiğini
    # yeniden kullanır ama orkestrasyon burada).
    shape_sess = BridgeShapeSession()

    # Liveness arming — HAND_FLIP ve PEEK_A_BOO'nun yanlış-pozitiflerini engeller.
    # • HAND_FLIP: ilk frame'de baseline_z alıyor; el kameraya yeni girdiğinde
    #   gürültüsü 0.03'ü aşıp anında tetikleniyor.
    # • PEEK_A_BOO: phase 0'da el yoksa phase 1'e geçiyor; kart açıldığında
    #   kullanıcı eli henüz görünmemişse phase 1'e hemen geçer, ilk el
    #   gösteriminde başarı sayar.
    # Çözüm: bu komutlar aktifken kullanıcı eli kameraya gösterip stabil
    # tutana kadar liveness.update'i atlıyoruz. Stabil = el 1 sn kesintisiz görünür.
    liveness_arming = False             # şu an arming bekleniyor mu
    liveness_arm_hand_since: float | None = None  # ilk hand görüldüğü an
    LIVENESS_ARM_HANDS_NEEDED = 1.0     # saniye
    gm = GameManager()
    # Tracker'ı GameManager'ın başlangıç moduna göre kur.
    tracker = _make_tracker_for(gm.current_mode)
    tracker_mode = gm.current_mode
    # Bu bağlantı için tercih edilen liveness komutu (kart bazlı odak).
    preferred_cmd: str | None = None
    fps_value = 0.0
    fps_prev = time.time()

    # ── Yüz görevleri ───────────────────────────────────────────────────
    # active_kind: "hand" (el modları) veya "face" (yüz görevleri). Aynı anda
    # tek tracker çalışır (ikisini birden işlemek pahalı). face_tracker tembel
    # oluşturulur (ilk yüz görevinde).
    active_kind = "hand"
    face_tracker: FaceTracker | None = None
    face_sess = FaceTaskSession()

    try:
        await ws.send_json({"type": "ready", "modes": MODE_NAMES})

        # Her bağlantı kendi tracker/GameManager örneğine sahip (oturum izolasyonu).
        # tracker'ı dıştaki finally kapatır; ayrıca şekil moduna geçişte tracker
        # tek-elli olarak yeniden kurulabildiği için 'with' yerine manuel yönetim.
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "set_face_task":
                # Yüz görevine geç (el modlarından ayrı izole akış).
                task_id = msg.get("task") or "blink"
                if task_id not in VALID_FACE_TASKS:
                    await ws.send_json({"type": "error",
                                        "message": f"Bilinmeyen yüz görevi: {task_id}"})
                    continue
                active_kind = "face"
                if face_tracker is None:
                    face_tracker = FaceTracker()
                face_sess.set_task(task_id)
                await ws.send_json({
                    "type": "face_task_set",
                    "task": task_id,
                })
                continue

            if kind == "set_mode":
                active_kind = "hand"
                target = int(msg.get("mode", 0))
                target = max(0, min(target, len(MODE_NAMES) - 1))
                # Kart bazlı liveness komut tercihi (örn. "WAVE"); yoksa None.
                preferred_cmd = msg.get("prefer_cmd") or None
                # GameManager sadece cycle_mode sunduğu için döngüsel ilerletiyoruz.
                safety = 0
                while gm.current_mode != target and safety < len(MODE_NAMES) + 1:
                    gm.cycle_mode()
                    safety += 1
                # Tracker'ı yeni modun ihtiyaçlarına göre yeniden kur (num_hands,
                # max_lost ve tracking_confidence moda göre farklı).
                if gm.current_mode != tracker_mode:
                    try:
                        tracker.close()
                    except Exception:
                        pass
                    tracker = _make_tracker_for(gm.current_mode)
                    tracker_mode = gm.current_mode
                # Şekil moduna girişte yeni bir tur (yeni rastgele şekil).
                if gm.current_mode == 5:
                    shape_sess.reset()
                # Sıralı'ya girince çizim adımlarını hemen atla.
                if gm.current_mode == 3:
                    _skip_drawing_seq_steps(gm.sequential)
                # Liveness modunda istenen komuta odakla.
                biased = False
                if gm.current_mode == 2 and preferred_cmd:
                    biased = _bias_liveness_command(gm, preferred_cmd)
                # HAND_FLIP / PEEK_A_BOO için arming gerekli (yanlış-pozitif önleme)
                liveness_arming = (gm.current_mode == 2 and
                                   preferred_cmd in ("HAND_FLIP", "PEEK_A_BOO"))
                liveness_arm_hand_since = None
                await ws.send_json({
                    "type": "mode_changed",
                    "mode": gm.current_mode,
                    "mode_name": MODE_NAMES[gm.current_mode],
                    "preferred_cmd": preferred_cmd,
                    "biased": biased,
                })
                continue

            if kind == "restart":
                # Yüz görevindeyse onu sıfırla.
                if active_kind == "face":
                    face_sess.reset()
                    await ws.send_json({"type": "restarted", "kind": "face",
                                        "task": face_sess.task_id})
                    continue
                gm.restart()
                if gm.current_mode == 5:
                    shape_sess.reset()
                if gm.current_mode == 3:
                    _skip_drawing_seq_steps(gm.sequential)
                # Yeniden başlatınca da kartın komut tercihini koru.
                biased = False
                if gm.current_mode == 2 and preferred_cmd:
                    biased = _bias_liveness_command(gm, preferred_cmd)
                liveness_arming = (gm.current_mode == 2 and
                                   preferred_cmd in ("HAND_FLIP", "PEEK_A_BOO"))
                liveness_arm_hand_since = None
                await ws.send_json({"type": "restarted",
                                    "mode": gm.current_mode,
                                    "mode_name": MODE_NAMES[gm.current_mode],
                                    "preferred_cmd": preferred_cmd,
                                    "biased": biased})
                continue

            if kind == "cycle_eval":
                if gm.current_mode == 5 and hasattr(gm.shape_eval, "cycle_mode"):
                    gm.shape_eval.cycle_mode()
                continue

            if kind != "frame":
                await ws.send_json({"type": "error", "message": f"Bilinmeyen tip: {kind}"})
                continue

            frame = _decode_b64_jpeg(msg.get("data", ""))
            if frame is None:
                await ws.send_json({"type": "error", "message": "Frame çözümlenemedi."})
                continue

            # PythonProject1, main.py içinde cv2.flip(frame, 1) yapıyor — biz de
            # aynı pre-processing'i uyguluyoruz ki handedness mantığı bozulmasın.
            frame = cv2.flip(frame, 1)

            # ── Yüz görevi dalı (el modlarından tamamen ayrı) ──────────────
            if active_kind == "face":
                try:
                    face = face_tracker.process(frame) if face_tracker else None
                    face_sess.update(face)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Yüz frame işleme hatası")
                    await ws.send_json({"type": "error", "message": f"İşleme hatası: {exc}"})
                    continue
                now = time.time()
                dt = now - fps_prev
                if dt > 0:
                    fps_value = 0.7 * (1.0 / dt) + 0.3 * fps_value
                fps_prev = now
                await ws.send_json(serialize_face(face_sess, face, fps_value))
                continue

            try:
                hands = tracker.process(frame)
                if gm.current_mode == 5:
                    # Mode 5: robust bridge oturumu (GameManager'ı atla).
                    shape_sess.update(hands)
                else:
                    # Mode 3 (Sıralı): çizim adımlarını otomatik atla (UI'da
                    # şablon yok, kullanıcı için belirsiz).
                    if gm.current_mode == 3:
                        _skip_drawing_seq_steps(gm.sequential)
                    # Mode 2 + arming: kullanıcı eli 1 sn stabil görünene kadar
                    # liveness.update'i atla (HAND_FLIP/PEEK_A_BOO yanlış-pozitif önleme).
                    if liveness_arming:
                        now_w = time.time()
                        if hands:
                            if liveness_arm_hand_since is None:
                                liveness_arm_hand_since = now_w
                            elif now_w - liveness_arm_hand_since >= LIVENESS_ARM_HANDS_NEEDED:
                                liveness_arming = False  # armed, normal akışa geç
                        else:
                            # El kaybolursa sayacı sıfırla (yarım kalmasın)
                            liveness_arm_hand_since = None
                    if not liveness_arming:
                        gm.update(hands)
                    else:
                        # Sadece "el algılandı/algılanmadı" güncellemesi için
                        gm.hands_present = bool(hands)
            except Exception as exc:  # noqa: BLE001
                log.exception("Frame işleme hatası")
                await ws.send_json({"type": "error", "message": f"İşleme hatası: {exc}"})
                continue

            # FPS (EMA)
            now = time.time()
            dt = now - fps_prev
            if dt > 0:
                fps_value = 0.7 * (1.0 / dt) + 0.3 * fps_value
            fps_prev = now

            if gm.current_mode == 5:
                payload = serialize_bridge_shape(shape_sess, hands, fps_value)
            else:
                payload = serialize_state(gm, hands, fps_value)
                # Arming durumu UI'a "hazırlanıyor…" gösterimi için
                if gm.current_mode == 2 and liveness_arming:
                    sess = payload.get("session", {})
                    sess["arming"] = True
                    if liveness_arm_hand_since is None:
                        sess["arming_text"] = "Elinizi kameraya gösterin"
                        sess["arming_progress"] = 0.0
                    else:
                        prog = min(
                            (time.time() - liveness_arm_hand_since)
                            / LIVENESS_ARM_HANDS_NEEDED,
                            1.0,
                        )
                        sess["arming_text"] = "Sabit tutun…"
                        sess["arming_progress"] = float(prog)
            await ws.send_json(payload)

    except WebSocketDisconnect:
        log.info("WS bağlantı kapatıldı.")
    except Exception as exc:  # noqa: BLE001
        log.exception("WS hatası")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            tracker.close()
        except Exception:
            pass
        if face_tracker is not None:
            try:
                face_tracker.close()
            except Exception:
                pass


@app.exception_handler(Exception)
async def _global(_, exc):
    log.exception("Beklenmeyen hata")
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
