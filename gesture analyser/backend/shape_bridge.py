"""
Bridge-side Şekil Çizim Oturumu
================================
PythonProject1'in ShapeTracerSession'ı tek-frame'lik titremelere/el-kayıplarına
sert tepki verip (parmak bir an kapalı sanılınca, ya da el bir frame kaybolunca)
çizimi yarıda kesiyordu. Bu modül, kullanıcının istediği davranışı SAĞLAR:

  • Çizim süre dolana kadar sürer; sonunda değerlendirilir.
  • Erken bitirmek için kullanıcı YUMRUK yapar (işaret parmağını kapatır) —
    ama tek-frame titremesi değil, ÜST ÜSTE birkaç frame (FIST_HOLD) gerekir.
  • Çizim sırasında el bir an kaybolursa BİTİRMEZ; birkaç saniye bekler, el
    ekrana dönerse kaldığı yerden DEVAM eder. (Yalnızca süre dolması bitirir.)
  • Bitiş-noktasına yakınlık tetiği YOKTUR (erken "doğrulandı" olmaz).

PythonProject1'e HİÇ dokunmaz — yalnızca şablonları ve DTW matematiğini ithal
ederek yeniden kullanır (generate_random_shape, _resample, _centroid_normalise,
dtw_normalised_cost, hand_scale, is_finger_open, Finger, _LM).
"""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

import bridge  # noqa: F401  — PythonProject1'i sys.path'e ekler (yan etki)
from shape_tracer import (
    generate_random_shape, ShapeTemplate,
    _resample, _centroid_normalise, dtw_normalised_cost,
    _finger_near_point, START_RADIUS, ARM_RADIUS,
    DEFAULT_DRAW_TIME, DEFAULT_DTW_THRESH, DEFAULT_RESAMPLE_N,
    DEFAULT_MIN_HS, DEFAULT_POS_HOLD, DEFAULT_INSTRUCT_TIME,
)
from gesture_validator import hand_scale as _hs, is_finger_open, Finger, _LM


class BState(Enum):
    INSTRUCTING = auto()   # kısa talimat
    IDLE        = auto()   # parmağı BAŞLA noktasına götür
    POSITIONING = auto()   # BAŞLA'da sabit tut (pos_hold)
    TRACING     = auto()   # çiz (süreye kadar / yumrukla biter)
    VERIFIED    = auto()
    FAILED      = auto()


# Robustluk parametreleri (bridge'e özgü — PythonProject1 sabitleri değil)
_FIST_HOLD_FRAMES   = 14     # yumruğun "bitir" sayılması için ardışık frame
                             # (~0.47s @30fps; titreme/parmak-kıvrılma yanlış-pozitifini önler)
_LOST_GRACE_SECONDS = 4.0    # el kaybında bekleme süresi (bilgi amaçlı; süre
                             # dolana kadar zaten bitirmiyoruz)
_RESULT_PAUSE       = 2.5    # sonuç gösterim süresi, sonra yeni tur

# DTW kabul eşiği. Ölçülen gerçek maliyetler (despike sonrası):
#   • İyi/temiz çizim:  ~0.05-0.10 (üçgen 0.046, S-eğri 0.064)
#   • Daire (zor):       ~0.20-0.23
#   • Yanlış çizimler:   ~0.25-0.50
# Eşik 0.18: iyi şekiller rahat geçer, yanlış çizimler reddedilir.
# (Not: PythonProject1 varsayılanı 0.25 ham masaüstü feed'i için ayarlı.)
_SHAPE_DTW_THRESHOLD = 0.18

# Spike eşiği — el-kaybı/glitch zıplamalarını yoldan temizler.
# Normal çizim adımı ~0.05; >0.08 zıplaması yapay sayılır.
_SPIKE_DIST = 0.08
# Birden çok geçişle daha agresif temizlik (büyük spike chain'leri için).
_DESPIKE_PASSES = 3


def _despike_once(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Tek geçişte spike temizliği."""
    import math
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    for i in range(1, len(path) - 1):
        px, py = out[-1]
        cx, cy = path[i]
        nx, ny = path[i + 1]
        d_pc = math.hypot(cx - px, cy - py)
        d_cn = math.hypot(nx - cx, ny - cy)
        d_pn = math.hypot(nx - px, ny - py)
        if d_pc > _SPIKE_DIST and d_cn > _SPIKE_DIST and d_pn < _SPIKE_DIST * 1.5:
            continue  # spike — atla
        out.append((cx, cy))
    out.append(path[-1])
    return out


def _despike(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Çoklu-geçiş spike temizliği — büyük gap'lerden sonra oluşan spike
    zincirlerini (örn. takip yeniden kilitlendiğinde 2-3 frame ardarda atlayan
    noktalar) tek geçişin yakalayamadığı durumlar için."""
    p = list(path)
    for _ in range(_DESPIKE_PASSES):
        new = _despike_once(p)
        if len(new) == len(p):
            break
        p = new
    return p


class BridgeShapeSession:
    """Robust, süreye-kadar-çiz şekil oturumu (HUMAN_TEST eşdeğeri)."""

    def __init__(self,
                 time_limit: float = DEFAULT_DRAW_TIME,
                 dtw_threshold: float = _SHAPE_DTW_THRESHOLD,
                 resample_n: int = DEFAULT_RESAMPLE_N,
                 min_hand_scale: float = DEFAULT_MIN_HS,
                 position_hold: float = DEFAULT_POS_HOLD,
                 instruct_time: float = DEFAULT_INSTRUCT_TIME):
        self.time_limit     = time_limit
        self.dtw_threshold  = dtw_threshold
        self.resample_n     = resample_n
        self.min_hand_scale = min_hand_scale
        self.position_hold  = position_hold
        self.instruct_time  = instruct_time
        self.reset()

    # ── Tur yaşam döngüsü ────────────────────────────────────────────
    def reset(self) -> None:
        self.template: ShapeTemplate = generate_random_shape()
        self.state = BState.INSTRUCTING
        self._traced: list[tuple[float, float]] = []
        self._raw_buf: list[tuple[float, float]] = []
        self._instruct_start = time.time()
        self._position_start: Optional[float] = None
        self._draw_start: Optional[float] = None
        self._result_at: Optional[float] = None
        self._fist_frames = 0
        self._lost_since: Optional[float] = None
        self.similarity = 0.0       # 0..100
        self.dtw_cost = 0.0

    # ── Nokta kaydı (3-örnek MA, yakın-duplikat atla) — ShapeTracer ile aynı ──
    def _push_point(self, x: float, y: float) -> None:
        self._raw_buf.append((x, y))
        if len(self._raw_buf) > 3:
            self._raw_buf.pop(0)
        n = len(self._raw_buf)
        sx = sum(p[0] for p in self._raw_buf) / n
        sy = sum(p[1] for p in self._raw_buf) / n
        if self._traced:
            lx, ly = self._traced[-1]
            if abs(sx - lx) < 0.003 and abs(sy - ly) < 0.003:
                return
        self._traced.append((sx, sy))

    # ── Doğrulama (ShapeTracerSession._run_verification ile birebir) ──
    def _verify(self) -> None:
        if len(self._traced) < self.template.min_trace_points:
            self.state = BState.FAILED
            self.similarity = 0.0
            self.dtw_cost = float("inf")
            self._result_at = time.time()
            return
        # Yolu doğrulamadan önce takip glitch'lerinden (spike) temizle.
        cleaned = _despike(self._traced)
        t_rs = _resample(list(self.template.waypoints), self.resample_n)
        u_rs = _resample(cleaned, self.resample_n)
        t_n = _centroid_normalise(t_rs)
        u_n = _centroid_normalise(u_rs)

        # ── Yön / başlangıç-noktası bağımsız DTW ──────────────────────────
        # Standart DTW, kullanıcının yolunu şablona NOKTA-NOKTA hizaladığı için
        # ters yönde ya da farklı köşeden başlanan AYNI şekil yüksek maliyet
        # alır (gerçek bir kareyi 0.9'a kadar çıkardığını ölçtük). Çözüm:
        # kapalı şekillerde tüm dairesel kaydırmaları + ters yönü dene, en
        # küçüğünü kullan. Açık şekillerde sadece ters yönü dene.
        is_closed = self.template.start_point == self.template.end_point
        u_rev = list(reversed(u_n))
        candidates = [u_n, u_rev]
        if is_closed:
            n = len(u_n)
            # 8 farklı başlangıç noktası dene (her N/8'de bir kaydırma).
            for shift in range(1, n, max(1, n // 8)):
                candidates.append(u_n[shift:] + u_n[:shift])
                candidates.append(u_rev[shift:] + u_rev[:shift])

        cost = min(dtw_normalised_cost(t_n, c) for c in candidates)
        self.dtw_cost = cost
        self.similarity = max(0.0, min(1.0, 1.0 - cost / self.dtw_threshold)) * 100.0
        self.state = BState.VERIFIED if cost <= self.dtw_threshold else BState.FAILED
        self._result_at = time.time()

    # ── Ana güncelleme ───────────────────────────────────────────────
    def update(self, hands: list) -> BState:
        now = time.time()

        # Sonuç → bekle → yeni tur (auto-advance)
        if self.state in (BState.VERIFIED, BState.FAILED):
            if self._result_at and now - self._result_at >= _RESULT_PAUSE:
                self.reset()
            return self.state

        # Geçerli (yeterince yakın/büyük) tek el seç
        hand = None
        if hands:
            h0 = hands[0]
            if _hs(h0.landmarks) >= self.min_hand_scale:
                hand = h0

        # INSTRUCTING → IDLE
        if self.state == BState.INSTRUCTING:
            if now - self._instruct_start >= self.instruct_time:
                self.state = BState.IDLE
            return self.state

        if self.state == BState.IDLE:
            if hand is not None:
                lm = hand.landmarks
                tip = lm[_LM["INDEX_TIP"]]
                if (is_finger_open(lm, hand.handedness, Finger.INDEX) and
                        _finger_near_point(tip.x, tip.y, *self.template.start_point, START_RADIUS)):
                    self._position_start = now
                    self.state = BState.POSITIONING
            return self.state

        if self.state == BState.POSITIONING:
            if hand is None:
                self._position_start = None
                self.state = BState.IDLE
                return self.state
            lm = hand.landmarks
            tip = lm[_LM["INDEX_TIP"]]
            if not (is_finger_open(lm, hand.handedness, Finger.INDEX) and
                    _finger_near_point(tip.x, tip.y, *self.template.start_point, START_RADIUS)):
                self._position_start = None
                self.state = BState.IDLE
            elif now - self._position_start >= self.position_hold:
                # Çizimi başlat
                self._traced.clear()
                self._raw_buf.clear()
                self._draw_start = now
                self._fist_frames = 0
                self._lost_since = None
                self.state = BState.TRACING
                self._push_point(tip.x, tip.y)
            return self.state

        if self.state == BState.TRACING:
            # 1) Süre doldu → değerlendir (tek "otomatik" bitiş budur)
            if now - self._draw_start >= self.time_limit:
                self._verify()
                return self.state

            # 2) El kaybı → BİTİRME. Birkaç saniye bekle; el dönünce devam.
            if hand is None:
                if self._lost_since is None:
                    self._lost_since = now
                # Nokta ekleme yok, yumruk sayacı sıfırla (titreme sayılmasın).
                self._fist_frames = 0
                return self.state
            else:
                self._lost_since = None

            lm = hand.landmarks
            tip = lm[_LM["INDEX_TIP"]]
            index_open = is_finger_open(lm, hand.handedness, Finger.INDEX)

            if index_open:
                # Kaldığı yerden devam — yeni noktayı ekle.
                self._push_point(tip.x, tip.y)
                self._fist_frames = 0
            else:
                # Yumruk: yalnızca ÜST ÜSTE _FIST_HOLD_FRAMES + yeterli nokta → bitir.
                self._fist_frames += 1
                if (self._fist_frames >= _FIST_HOLD_FRAMES and
                        len(self._traced) >= self.template.min_trace_points):
                    self._verify()
            return self.state

        return self.state

    # ── HUD / serializer için gözlemlenebilir alanlar ────────────────
    @property
    def state_name(self) -> str:
        return self.state.name

    @property
    def shape_label(self) -> str:
        return self.template.label

    @property
    def traced_path(self) -> list[tuple[float, float]]:
        return list(self._traced)

    @property
    def template_waypoints(self) -> list[tuple[float, float]]:
        return list(self.template.waypoints)

    @property
    def point_count(self) -> int:
        return len(self._traced)

    @property
    def time_remaining(self) -> float:
        if self._draw_start is None:
            return self.time_limit
        return max(0.0, self.time_limit - (time.time() - self._draw_start))

    @property
    def position_progress(self) -> float:
        if self._position_start is None:
            return 0.0
        return min((time.time() - self._position_start) / self.position_hold, 1.0)

    @property
    def fist_progress(self) -> float:
        """0..1 — yumruğun bitirme eşiğine doğru dolması (HUD geri bildirimi)."""
        return min(self._fist_frames / _FIST_HOLD_FRAMES, 1.0)

    @property
    def hand_lost(self) -> bool:
        return self._lost_since is not None
