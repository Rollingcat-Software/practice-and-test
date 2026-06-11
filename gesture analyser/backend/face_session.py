"""
Yüz Görevi Oturumu — 14 baş/yüz hareketi
==========================================
PythonProject1'e dokunmadan, bridge tarafında yazılmış yüz canlılık görevleri.
FaceTracker'ın ürettiği blendshape skorları + landmark-tabanlı baş pozunu
kullanır. Nod/shake (baş salla) için PythonProject1'in WaveDetector'ünü yeniden
kullanır (osilasyon + frekans filtresi).

Görevler:
  blink        Göz Kırp            — iki göz kapat→aç (döngü)
  wink_left    Sol Gözü Kapat      — sol göz kapalı, sağ açık
  wink_right   Sağ Gözü Kapat      — sağ göz kapalı, sol açık
  smile        Geniş Gülümse       — mouthSmile
  mouth_open   Ağzı Aç             — jawOpen
  brows_up     Her İki Kaşı Kaldır — browInnerUp
  brow_left    Sol Kaşı Kaldır     — asimetrik (deneysel)
  brow_right   Sağ Kaşı Kaldır     — asimetrik (deneysel)
  head_left    Başı Sola Çevir     — yaw poz
  head_right   Başı Sağa Çevir     — yaw poz
  look_up      Yukarı Bak          — pitch poz
  look_down    Aşağı Bak           — pitch poz
  nod_yes      Başı Salla (Evet)   — pitch osilasyonu
  shake_no     Başı Salla (Hayır)  — yaw osilasyonu
"""
from __future__ import annotations

import os
import sys
import time
from enum import Enum, auto
from typing import Optional

# Bu dosyanın dizinini (backend) sys.path'e ekle — bridge importu çalışma
# dizinini PythonProject1'e çevirdiği için yerel modüller (face_tracker) aksi
# halde bulunamayabilir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bridge  # noqa: F401,E402 — sys.path'e PythonProject1'i ekler
from motion_analyzer import WaveDetector  # noqa: E402
from face_tracker import FaceResult  # noqa: E402


# ── Kalibrasyon sabitleri (canlı veriyle ayarlandı) ──────────────────
# Aynalı (flip) kare nedeniyle MediaPipe'ın blendshape sol/sağı kullanıcının
# gerçek sol/sağının TERSİ. Canlı ölçümle doğrulandı: sol göz kapatınca
# eyeBlinkRight, sol kaş kaldırınca browOuterUpRight yükseliyor → True.
BLENDSHAPE_SWAP_LR = True
# Baş pozu yaw/pitch işareti ters gelirse bu çarpanları -1 yap.
YAW_SIGN   = 1.0
PITCH_SIGN = 1.0

_HOLD_FRAMES   = 8       # ifade görevleri için ardışık frame (~0.4s)
_WINK_FRAMES   = 4       # wink için (daha kısa)
_ARM_SECONDS   = 0.6     # poz görevlerinde baseline yakalama süresi
_RESULT_HOLD   = 1.2     # SUCCESS gösterimi (sticky; restart ile sıfırlanır)

_BLINK_CLOSE   = 0.45    # göz "kapalı" eşiği (iki göz blink döngüsü için)
_BLINK_OPEN    = 0.30    # göz "açık" eşiği
# Wink (tek göz) — BASELINE-GÖRELİ: hedef göz blink'i nötrün üstüne yükselsin
# (_WINK_DELTA) ve diğer gözden belirgin yüksek olsun (_WINK_ABS). Eli o gözün
# üstüne koymak da o gözü "kapalı" gösterdiğinden doğal çalışır.
# (Canlı veri: sol göz Δ+0.265, sağ göz Δ+0.155.)
# ÖNEMLİ: mutlak (target-other) kıyası işe yaramaz çünkü eyeBlinkLeft nötrde
# bile eyeBlinkRight'tan ~0.17 yüksek. Bunun yerine İKİ gözün de kendi nötründen
# YÜKSELİŞİNİ kıyaslıyoruz (delta-gap): hedef gözün yükselişi diğerininkinden
# belirgin fazla olmalı.
_WINK_DELTA    = 0.13    # hedef gözün nötr üstü minimum yükselişi
_WINK_GAP      = 0.10    # (hedef yükselişi) - (diğer yükselişi) minimum farkı
_SMILE_THR     = 0.50    # gülümseme (küçük gülüşü kabul etmesin diye yükseltildi)
_MOUTH_THR     = 0.40
_BROW_THR      = 0.40    # iki kaş birden (browInnerUp, swap'sız)
# Asimetrik kaş (tek kaş) — BASELINE-GÖRELİ delta-gap. (Canlı veri: sol kaş Δ+0.546.)
_BROW_DELTA    = 0.20    # hedef kaşın nötr üstü minimum yükselişi
_BROW_GAP      = 0.15    # (hedef yükselişi) - (diğer yükselişi) minimum farkı


class FaceState(Enum):
    NO_FACE = auto()
    ARMING  = auto()      # baseline yakalanıyor (poz görevleri)
    ACTIVE  = auto()
    SUCCESS = auto()


# Görev tanımları: kind + parametreler
_TASKS: dict[str, dict] = {
    "blink":      {"kind": "blink"},
    "wink_left":  {"kind": "wink", "eye": "left"},
    "wink_right": {"kind": "wink", "eye": "right"},
    "smile":      {"kind": "expr", "bs": ["mouthSmileLeft", "mouthSmileRight"], "thr": _SMILE_THR},
    "mouth_open": {"kind": "expr", "bs": ["jawOpen"], "thr": _MOUTH_THR},
    "brows_up":   {"kind": "expr", "bs": ["browInnerUp"], "thr": _BROW_THR},
    "brow_left":  {"kind": "brow_asym", "side": "left"},
    "brow_right": {"kind": "brow_asym", "side": "right"},
    "head_left":  {"kind": "pose", "axis": "yaw",   "dir": -1, "thr": 0.35},
    "head_right": {"kind": "pose", "axis": "yaw",   "dir": +1, "thr": 0.35},
    "look_up":    {"kind": "pose", "axis": "pitch", "dir": -1, "thr": 0.12},
    "look_down":  {"kind": "pose", "axis": "pitch", "dir": +1, "thr": 0.12},
    "nod_yes":    {"kind": "osc",  "axis": "pitch"},
    "shake_no":   {"kind": "osc",  "axis": "yaw"},
}

VALID_FACE_TASKS = set(_TASKS.keys())


def _eye_blinks(bs: dict[str, float]) -> tuple[float, float]:
    """(sol, sağ) göz kırpma skoru — swap uygulanmış."""
    l = bs.get("eyeBlinkLeft", 0.0)
    r = bs.get("eyeBlinkRight", 0.0)
    return (r, l) if BLENDSHAPE_SWAP_LR else (l, r)


def _brow_ups(bs: dict[str, float]) -> tuple[float, float]:
    """(sol, sağ) dış kaş kaldırma skoru — swap uygulanmış."""
    l = bs.get("browOuterUpLeft", 0.0)
    r = bs.get("browOuterUpRight", 0.0)
    return (r, l) if BLENDSHAPE_SWAP_LR else (l, r)


class FaceTaskSession:
    """Tek bir yüz görevini sürdüren state machine."""

    def __init__(self, task_id: str = "blink"):
        self.set_task(task_id)

    def set_task(self, task_id: str) -> None:
        if task_id not in _TASKS:
            task_id = "blink"
        self.task_id = task_id
        self.task = _TASKS[task_id]
        self.reset()

    def reset(self) -> None:
        kind = self.task["kind"]
        self.state = FaceState.NO_FACE
        self._hold = 0
        self._phase = 0            # blink/wink döngü fazı
        self._success_at: Optional[float] = None
        # baseline (poz: yaw/pitch; wink: göz L/R; kaş: kaş L/R)
        self._arm_start: Optional[float] = None
        self._base0 = 0.0   # poz: yaw  | wink/kaş: kanal 0 (sol-etiket)
        self._base1 = 0.0   # poz: pitch| wink/kaş: kanal 1 (sağ-etiket)
        self._arm_samples: list[tuple[float, float]] = []
        # canlı metrikler (HUD)
        self.metric_value = 0.0
        self.metric_target = 1.0
        self.progress = 0.0
        # osilasyon dedektörü
        if kind == "osc":
            if self.task["axis"] == "pitch":   # nod (yukarı-aşağı, küçük genlik)
                self._wave = WaveDetector(buffer_size=30, min_swing=0.04,
                                          min_total_displacement=0.10, min_reversals=2,
                                          min_freq_hz=0.4, max_freq_hz=3.0)
            else:                              # shake (sağ-sol, büyük genlik)
                self._wave = WaveDetector(buffer_size=30, min_swing=0.10,
                                          min_total_displacement=0.28, min_reversals=2,
                                          min_freq_hz=0.4, max_freq_hz=3.0)
        else:
            self._wave = None

    # ── Ana güncelleme ───────────────────────────────────────────────
    def update(self, face: Optional[FaceResult]) -> FaceState:
        now = time.time()

        if self.state == FaceState.SUCCESS:
            return self.state  # sticky — restart ile sıfırlanır

        if face is None:
            # Yüz yok → ilerlemeyi sıfırla, bekle
            self.state = FaceState.NO_FACE
            self._hold = 0
            self._phase = 0
            self._arm_start = None
            self._arm_samples.clear()
            if self._wave is not None:
                self._wave.reset()
            self.progress = 0.0
            return self.state

        kind = self.task["kind"]
        # poz, wink ve asimetrik kaş — hepsi nötr baseline gerektirir (sinyaller
        # sıfır değil ve kişiye göre asimetrik). osc ve diğerleri gerektirmez.
        needs_arm = kind in ("pose", "wink", "brow_asym")

        # Yüz yeni geldi → uygun başlangıç durumuna geç
        if self.state == FaceState.NO_FACE:
            if needs_arm:
                self.state = FaceState.ARMING
                self._arm_start = now
                self._arm_samples.clear()
            else:
                self.state = FaceState.ACTIVE

        # ── ARMING (nötr baseline yakala) ─────────────────────────────
        if self.state == FaceState.ARMING:
            ch0, ch1 = self._arm_channels(face)
            self._arm_samples.append((ch0, ch1))
            self.progress = min((now - (self._arm_start or now)) / _ARM_SECONDS, 1.0)
            if now - (self._arm_start or now) >= _ARM_SECONDS and self._arm_samples:
                c0 = [s[0] for s in self._arm_samples]
                c1 = [s[1] for s in self._arm_samples]
                self._base0 = sum(c0) / len(c0)
                self._base1 = sum(c1) / len(c1)
                self.state = FaceState.ACTIVE
                self.progress = 0.0
            return self.state

        # ── ACTIVE: göreve göre dedektör ──────────────────────────────
        if kind == "blink":
            self._detect_blink(face)
        elif kind == "wink":
            self._detect_wink(face)
        elif kind == "expr":
            self._detect_expr(face)
        elif kind == "brow_asym":
            self._detect_brow_asym(face)
        elif kind == "pose":
            self._detect_pose(face)
        elif kind == "osc":
            self._detect_osc(face)

        return self.state

    def _succeed(self) -> None:
        self.state = FaceState.SUCCESS
        self._success_at = time.time()
        self.progress = 1.0

    def _arm_channels(self, face: FaceResult) -> tuple[float, float]:
        """Baseline yakalama için (kanal0, kanal1) döndürür — göreve göre."""
        kind = self.task["kind"]
        if kind == "pose":
            return face.yaw_norm, face.pitch_norm
        if kind == "wink":
            return _eye_blinks(face.blendshapes)     # (sol-etiket, sağ-etiket)
        if kind == "brow_asym":
            return _brow_ups(face.blendshapes)        # (sol-etiket, sağ-etiket)
        return 0.0, 0.0

    # ── Dedektörler ──────────────────────────────────────────────────
    def _detect_blink(self, face: FaceResult) -> None:
        l, r = _eye_blinks(face.blendshapes)
        self.metric_value = (l + r) / 2.0
        self.metric_target = _BLINK_CLOSE
        both_closed = l > _BLINK_CLOSE and r > _BLINK_CLOSE
        both_open = l < _BLINK_OPEN and r < _BLINK_OPEN
        if self._phase == 0:
            self.progress = min(self.metric_value / _BLINK_CLOSE, 1.0)
            if both_closed:
                self._phase = 1
        elif self._phase == 1:
            self.progress = 1.0
            if both_open:
                self._succeed()

    def _detect_wink(self, face: FaceResult) -> None:
        ch0, ch1 = _eye_blinks(face.blendshapes)   # (sol-etiket, sağ-etiket)
        if self.task["eye"] == "left":
            target, other, bt, bo = ch0, ch1, self._base0, self._base1
        else:
            target, other, bt, bo = ch1, ch0, self._base1, self._base0
        dt = target - bt          # hedef gözün nötr üstü yükselişi
        do = other - bo           # diğer gözün nötr üstü yükselişi
        self.metric_value = dt
        self.metric_target = _WINK_DELTA
        # Hedef göz nötrden belirgin yükselsin VE diğerinden çok daha fazla
        # yükselsin (her ikisi de kendi baseline'ına göre).
        cond = dt > _WINK_DELTA and (dt - do) > _WINK_GAP
        if cond:
            self._hold += 1
            self.progress = min(self._hold / _WINK_FRAMES, 1.0)
            if self._hold >= _WINK_FRAMES:
                self._succeed()
        else:
            self._hold = 0
            self.progress = max(0.0, min(dt / _WINK_DELTA, 1.0))

    def _detect_expr(self, face: FaceResult) -> None:
        vals = [face.blendshapes.get(b, 0.0) for b in self.task["bs"]]
        v = sum(vals) / len(vals)
        thr = self.task["thr"]
        self.metric_value = v
        self.metric_target = thr
        if v > thr:
            self._hold += 1
            self.progress = min(self._hold / _HOLD_FRAMES, 1.0)
            if self._hold >= _HOLD_FRAMES:
                self._succeed()
        else:
            self._hold = 0
            self.progress = min(v / thr, 1.0)

    def _detect_brow_asym(self, face: FaceResult) -> None:
        ch0, ch1 = _brow_ups(face.blendshapes)     # (sol-etiket, sağ-etiket)
        if self.task["side"] == "left":
            target, other, bt, bo = ch0, ch1, self._base0, self._base1
        else:
            target, other, bt, bo = ch1, ch0, self._base1, self._base0
        dt = target - bt          # hedef kaşın nötr üstü yükselişi
        do = other - bo           # diğer kaşın nötr üstü yükselişi
        self.metric_value = dt
        self.metric_target = _BROW_DELTA
        # Hedef kaş nötrden belirgin kalksın VE diğerinden çok daha fazla kalksın.
        cond = dt > _BROW_DELTA and (dt - do) > _BROW_GAP
        if cond:
            self._hold += 1
            self.progress = min(self._hold / _HOLD_FRAMES, 1.0)
            if self._hold >= _HOLD_FRAMES:
                self._succeed()
        else:
            self._hold = 0
            self.progress = max(0.0, min(dt / _BROW_DELTA, 1.0))

    def _detect_pose(self, face: FaceResult) -> None:
        axis = self.task["axis"]
        if axis == "yaw":
            signal = (face.yaw_norm - self._base0) * YAW_SIGN   # base0 = yaw
        else:
            signal = (face.pitch_norm - self._base1) * PITCH_SIGN  # base1 = pitch
        directed = signal * self.task["dir"]   # doğru yönde pozitif olmalı
        thr = self.task["thr"]
        self.metric_value = directed
        self.metric_target = thr
        if directed > thr:
            self._hold += 1
            self.progress = min(self._hold / _HOLD_FRAMES, 1.0)
            if self._hold >= _HOLD_FRAMES:
                self._succeed()
        else:
            self._hold = 0
            self.progress = max(0.0, min(directed / thr, 1.0))

    def _detect_osc(self, face: FaceResult) -> None:
        axis = self.task["axis"]
        val = face.pitch_norm if axis == "pitch" else face.yaw_norm
        self._wave.push(val)
        self.metric_value = float(self._wave.reversal_count)
        self.metric_target = float(self._wave.min_reversals)
        self.progress = min(self._wave.reversal_count / max(self._wave.min_reversals, 1), 1.0)
        if self._wave.is_waving():
            self._succeed()

    # ── HUD için ─────────────────────────────────────────────────────
    @property
    def state_name(self) -> str:
        return self.state.name
