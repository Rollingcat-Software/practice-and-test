"""
Gesture Analyser köprüsü
========================
El takip motoru ("Gesture Analyser") dizinini sys.path'e ekler — kaynak
dosyalara hiç dokunmadan HandTracker, GameManager ve oturum modüllerini
ithal eder. (Bu motorun eski adı "PythonProject1" idi; her iki ad da desteklenir.)

Motor konumu şu sırayla aranır (taşınabilirlik için):
  1. GESTURE_ANALYSER_PATH / PYTHONPROJECT1_PATH ortam değişkeni (en yüksek öncelik)
  2. backend yanında ("Gesture Analyser" veya "PythonProject1")
  3. <proje kökü> (backend ve frontend ile AYNI dizin)
  4. bir üst dizin
  5. geliştirme makinesi yolu (en düşük öncelik)

Yani motor klasörünü ("Gesture Analyser") projenin köküne (backend/ ve frontend/
yanına) koymak yeterli — başka ayar gerekmez.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # .../backend

# Motor klasörünün olası adları (yeni ad öncelikli, eski ad geriye uyumluluk).
_ENGINE_NAMES = ["Gesture Analyser", "PythonProject1"]
# Aranacak temel konumlar.
_ENGINE_BASES = [_HERE, _HERE.parent, _HERE.parent.parent]


def _looks_like_engine(p: Path) -> bool:
    """Bir dizinin gerçekten el takip motoru olup olmadığını doğrular."""
    return (p / "hand_tracker.py").exists() and (p / "hand_landmarker.task").exists()


# Aday listesini kur: önce ortam değişkeni, sonra konum×ad kombinasyonları,
# en sonda geliştirme makinesi yedeği.
_CANDIDATES: list = []
_env = os.environ.get("GESTURE_ANALYSER_PATH") or os.environ.get("PYTHONPROJECT1_PATH")
if _env:
    _CANDIDATES.append(_env)
for _base in _ENGINE_BASES:
    for _name in _ENGINE_NAMES:
        _CANDIDATES.append(_base / _name)
_CANDIDATES.append(r"C:\Users\hp\PycharmProjects\PythonProject1")  # geliştirme yedeği

PROJECT_PATH = None
for _cand in _CANDIDATES:
    if not _cand:
        continue
    _p = Path(_cand).resolve()
    if _p.exists() and _looks_like_engine(_p):
        PROJECT_PATH = _p
        break

if PROJECT_PATH is None:
    _searched = "\n  - ".join(str(Path(c).resolve()) for c in _CANDIDATES if c)
    raise RuntimeError(
        "El takip motoru ('Gesture Analyser') bulunamadı. Klasörü projenin "
        "köküne (backend/ ve frontend/ yanına) koyun ya da GESTURE_ANALYSER_PATH "
        "ortam değişkenini ayarlayın.\nAranan konumlar:\n  - " + _searched
    )

# PythonProject1 modülleri kendi aralarında göreli olmayan ithallar kullanıyor
# (ör: `from hand_tracker import ...`). Bu yüzden dizini sys.path'e ekliyoruz.
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

# Çalışma dizinini PythonProject1'e çeviriyoruz çünkü app_config göreli
# `config.yaml` yolu, hand_tracker da göreli `hand_landmarker.task` yolu okuyor.
os.chdir(str(PROJECT_PATH))

# --- PythonProject1 modülleri (kaynak değiştirilmedi) ----------------------
from hand_tracker import HandTracker, HandResult, draw_landmarks  # noqa: E402
from game_manager import GameManager, MODE_NAMES                  # noqa: E402
from gesture_session import SessionState                          # noqa: E402
from math_session import MathState                                # noqa: E402
from liveness_session import LivenessState, CmdType               # noqa: E402
from sequential_session import SeqState                           # noqa: E402
from finger_touch_session import TouchTestState                   # noqa: E402
from shape_tracer import TracerState                              # noqa: E402
from shape_trace_eval_session import EvalMode                     # noqa: E402

__all__ = [
    "HandTracker", "HandResult", "draw_landmarks",
    "GameManager", "MODE_NAMES",
    "SessionState", "MathState", "LivenessState", "CmdType",
    "SeqState", "TouchTestState", "TracerState", "EvalMode",
    "PROJECT_PATH",
]
