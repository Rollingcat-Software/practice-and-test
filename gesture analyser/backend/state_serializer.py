"""
GameManager durum serileştirici
================================
GameManager'ın iç durumunu (her mod için ayrı), tarayıcının render edebileceği
JSON-uyumlu sözlüğe çevirir.

PythonProject1 kaynak modüllerinde HİÇ değişiklik yapmaz — sadece public
attribute / property erişimini kullanır.
"""
from __future__ import annotations

from typing import Any

from bridge import (
    GameManager, MODE_NAMES, HandResult,
    SessionState, MathState, LivenessState,
    SeqState, TouchTestState,
)


def _hand_to_json(h: HandResult) -> dict[str, Any]:
    return {
        "handedness": h.handedness,
        "landmarks": [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in h.landmarks
        ],
    }


def _gesture_dict(gm: GameManager) -> dict[str, Any]:
    s = gm.gesture
    return {
        "mode_key": "gesture",
        "state": s.state.name,
        "target": s.current_target,
        "score": s.score,
        "hold_progress": float(s.hold_progress),
        "display_text": s.display_text,
        "status_label": s.status_label,
        "last_total": s._last_total,  # public-ish — okunuyor, değiştirilmiyor
    }


def _math_dict(gm: GameManager) -> dict[str, Any]:
    """math_session.MathSession alanlarını UI'ya çıkarır.

    Gerçek attribute isimleri (PythonProject1/math_session.py):
        state (MathState), score, equation_text, answer, time_remaining,
        hold_progress, display_text, status_label
    """
    s = gm.math
    # detected_total() bir method — landmark'lardan dönüyor; son frame'i
    # GameManager.update sırasında hesapladığı için doğrudan çağırabiliriz.
    try:
        last_total = s.detected_total()
    except Exception:
        last_total = None

    return {
        "mode_key":         "math",
        "state":            s.state.name,
        "score":            int(s.score),
        "equation_text":    s.equation_text,
        "time_remaining":   float(s.time_remaining),
        "hold_progress":    float(s.hold_progress),
        "display_text":     s.display_text,
        "status_label":     s.status_label,
        "last_total":       last_total,
        # answer (beklenen cevap) — debug için yararlı ama UI'da gizliyoruz.
        "_answer_debug":    int(s.answer),
    }


def _wave_debug(liveness) -> dict[str, Any]:
    """WaveDetector iç buffer'ından okuma — yalnızca teşhis için.

    Kullanılan attribute'lar (motion_analyzer.WaveDetector):
        _data: deque[(time, x)]
        buffer_size, min_total_disp, min_swing, min_reversals,
        min_freq, max_freq
    """
    try:
        w = liveness._wave
        data = list(w._data)
        xs = [d[1] for d in data]
        ts = [d[0] for d in data]
        total = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
        span = (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
        return {
            "wave_buffer_len":   len(data),
            "wave_buffer_max":   int(w.buffer_size),
            "wave_total_disp":   float(total),
            "wave_min_total":    float(w.min_total_disp),
            "wave_min_swing":    float(w.min_swing),
            "wave_min_reversals": int(w.min_reversals),
            "wave_min_freq":     float(w.min_freq),
            "wave_max_freq":     float(w.max_freq),
            "wave_span_s":       float(span),
            "wave_last_x":       float(xs[-1]) if xs else None,
        }
    except Exception:
        return {}


def _liveness_dict(gm: GameManager) -> dict[str, Any]:
    """LivenessChallenge alanlarını UI'ya çıkarır.

    Gerçek attribute / property isimleri (PythonProject1/liveness_session.py):
        state, current_cmd (Command: .name, .cmd_type),
        score, streak, challenges_completed, num_challenges,
        time_remaining, debounce_progress, verification_pct,
        wave_reversals, area_change_pct,
        is_spoof_blocked, spoof_warning_progress,
        is_wave_cmd, is_touch_cmd, is_shape_trace_cmd,
        touch_frame_count, touch_frame_progress,
        display_text, status_label, command_label,
        challenge_progress_text, shape_trace_label
    """
    s = gm.liveness

    cmd_name = ""
    cmd_type = ""
    if s.current_cmd is not None:
        cmd_name = s.current_cmd.name
        cmd_type = s.current_cmd.cmd_type.name

    return {
        "mode_key":             "liveness",
        "state":                s.state.name,
        "command_name":         cmd_name,         # Örn: "WAVE YOUR HAND!"
        "command_type":         cmd_type,         # Örn: "WAVE", "GESTURE", "HAND_FLIP"
        "score":                int(s.score),
        "streak":               int(s.streak),
        "challenges_completed": int(s.challenges_completed),
        "num_challenges":       int(s.num_challenges),
        "verification_pct":     float(s.verification_pct),
        "time_remaining":       float(s.time_remaining),
        "debounce_progress":    float(s.debounce_progress),
        "wave_reversals":       int(s.wave_reversals),
        # WAVE debug: buffer doluluğu + toplam yatay deplasman + son x
        # (PythonProject1'in iç state'i — okuma amaçlı, yazmıyoruz)
        **_wave_debug(s),
        "area_change_pct":      (None if s.area_change_pct is None
                                 else float(s.area_change_pct)),
        "is_spoof_blocked":     bool(s.is_spoof_blocked),
        "spoof_warning_progress": float(s.spoof_warning_progress),
        "is_wave_cmd":          bool(s.is_wave_cmd),
        "is_touch_cmd":         bool(s.is_touch_cmd),
        "is_shape_trace_cmd":   bool(s.is_shape_trace_cmd),
        "touch_frame_count":    int(s.touch_frame_count),
        "touch_frame_progress": float(s.touch_frame_progress),
        "display_text":         s.display_text,
        "status_label":         s.status_label,
        "command_label":        (s.command_label if s.current_cmd else ""),
        "challenge_progress":   s.challenge_progress_text,
    }


def _sequential_dict(gm: GameManager) -> dict[str, Any]:
    """SequentialSession alanlarını UI'ya çıkarır.

    Gerçek property isimleri (PythonProject1/sequential_session.py):
        state (SeqState), current_step (SeqStep|None), current_step_idx,
        total_steps, passed_count, step_results, hold_progress,
        step_time_remaining, step_time_elapsed, display_text, status_label,
        progress_text, step.name, step.step_type, step.target
    """
    s = gm.sequential
    step = s.current_step
    return {
        "mode_key":      "sequential",
        "state":         s.state.name,
        "current_idx":   int(s.current_step_idx),
        "total_steps":   int(s.total_steps),
        "passed_count":  int(s.passed_count),
        # Mevcut adım bilgisi
        "step_name":     step.name if step else "",
        "step_type":     step.step_type if step else "",
        "step_target":   int(step.target) if step else 0,
        # Zamanlama / ilerleme
        "time_remaining": float(s.step_time_remaining),
        "step_time_limit": float(step.time_limit) if step else 0.0,
        "hold_progress": float(s.hold_progress),
        # Sonuç listesi (her adım için PENDING/PASSED/TIMED_OUT)
        "results":       [r.name for r in s.step_results],
        # Native metinler
        "display_text":  s.display_text,
        "status_label":  s.status_label,
        "progress_text": s.progress_text,
    }


def _touch_test_dict(gm: GameManager) -> dict[str, Any]:
    """FingerTouchSession alanlarını UI'ya çıkarır.

    Gerçek property isimleri (PythonProject1/finger_touch_session.py):
        state (TouchTestState), current_idx, passed (list[bool]),
        passed_count, command_label, command_name, progress_text,
        frame_count, hold_progress, all_commands
    """
    s = gm.touch_test
    # all_commands: (TouchCommand, etiket, geçti_mi) — checklist için
    commands = []
    try:
        for cmd, label, passed in s.all_commands:
            commands.append({
                "name": cmd.name,
                "label": label,
                "passed": bool(passed),
            })
    except Exception:
        pass

    return {
        "mode_key":      "touch_test",
        "state":         s.state.name,
        "current_idx":   int(s.current_idx),
        "passed_count":  int(s.passed_count),
        "total":         len(commands) or 5,
        "command_label": s.command_label,   # "PINCH: THUMB + INDEX   (IDs 4 & 8)"
        "command_name":  s.command_name,    # "THUMB_TO_INDEX"
        "progress_text": s.progress_text,   # "Command 1 / 5"
        "frame_count":   int(s.frame_count),
        "verify_frames": int(getattr(s, "verify_frames", 10)),
        "hold_progress": float(s.hold_progress),
        "commands":      commands,
    }


def _round_path(path, nd: int = 4):
    """[(x,y), ...] → [[x,y], ...] yuvarlanmış, JSON-uyumlu."""
    out = []
    for p in path:
        try:
            out.append([round(float(p[0]), nd), round(float(p[1]), nd)])
        except Exception:
            pass
    return out


def _shape_eval_dict(gm: GameManager) -> dict[str, Any]:
    """ShapeTraceEvalSession (HUMAN_TEST) alanlarını UI'ya çıkarır.

    Gerçek property isimleri (PythonProject1/shape_trace_eval_session.py
    + shape_tracer.py):
        eval_mode (EvalMode.label), eval_state, shape_label, state_label,
        human_tracer_state (TracerState), human_time_remaining,
        human_position_progress, human_point_count,
        debug_template_path, debug_user_path, current_template
        (.start_point, .end_point, .min_trace_points),
        result_similarity, result_dtw_cost, result_drift, result_time
    """
    s = gm.shape_eval
    tmpl = s.current_template

    tracer_state = s.human_tracer_state
    tracer_state_name = tracer_state.name if tracer_state is not None else None

    # INSTRUCTING geri sayımı + tracing draw_progress için iç tracer (okuma)
    instruct_remaining = 0.0
    draw_progress = 0.0
    try:
        tr = s._tracer
        if tr is not None:
            instruct_remaining = float(tr.instruct_remaining)
            draw_progress = float(tr.draw_progress)
    except Exception:
        pass

    return {
        "mode_key":          "shape_eval",
        "eval_mode":         s.eval_mode.label,        # "HUMAN TEST"
        "eval_state":        s.eval_state.name,        # IDLE / RUNNING / RESULT
        "shape_label":       s.shape_label,            # CIRCLE / SQUARE / TRIANGLE / S-CURVE
        "state_label":       s.state_label,            # tracer state / sonuç / "SIMULATING %"
        "tracer_state":      tracer_state_name,        # INSTRUCTING/IDLE/POSITIONING/TRACING/VERIFIED/FAILED
        "time_remaining":    float(s.human_time_remaining),
        "position_progress": float(s.human_position_progress),
        "draw_progress":     draw_progress,
        "instruct_remaining": instruct_remaining,
        "point_count":       int(s.human_point_count),
        "min_trace_points":  int(tmpl.min_trace_points) if tmpl else 0,
        # Sonuç metrikleri (RESULT'ta anlamlı)
        "similarity":        float(s.result_similarity),
        "dtw_cost":          float(s.result_dtw_cost),
        "result_time":       float(s.result_time),
        # Görsel overlay yolları (normalize [0,1] koordinat)
        "template_path":     _round_path(s.debug_template_path),
        "user_path":         _round_path(s.debug_user_path),
        "start_point":       ([round(tmpl.start_point[0], 4), round(tmpl.start_point[1], 4)]
                              if tmpl else None),
        "end_point":         ([round(tmpl.end_point[0], 4), round(tmpl.end_point[1], 4)]
                              if tmpl else None),
    }


_DISPATCH = {
    0: _gesture_dict,
    1: _math_dict,
    2: _liveness_dict,
    3: _sequential_dict,
    4: _touch_test_dict,
    5: _shape_eval_dict,
}


def serialize_state(gm: GameManager, hands: list[HandResult], fps: float) -> dict[str, Any]:
    mode_idx = gm.current_mode
    session = _DISPATCH[mode_idx](gm)
    return {
        "type": "state",
        "mode": mode_idx,
        "mode_name": MODE_NAMES[mode_idx],
        "hands_present": bool(gm.hands_present),
        "hands": [_hand_to_json(h) for h in hands],
        "session": session,
        "fps": round(float(fps), 1),
    }


def _face_group(lm, idxs):
    return [[round(float(lm[i].x), 4), round(float(lm[i].y), 4)] for i in idxs]


def serialize_face(session, face, fps: float) -> dict[str, Any]:
    """FaceTaskSession + FaceResult durumunu UI'ya serileştirir (mode_key 'face').

    Yüz mesh'i için tam 478 nokta yerine kontur alt kümeleri gönderilir (hafif).
    """
    from face_tracker import (
        FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS_OUTER, LEFT_BROW, RIGHT_BROW,
    )

    present = face is not None
    groups: dict[str, Any] = {}
    if present:
        lm = face.landmarks
        groups = {
            "oval":       _face_group(lm, FACE_OVAL),
            "left_eye":   _face_group(lm, LEFT_EYE),
            "right_eye":  _face_group(lm, RIGHT_EYE),
            "lips":       _face_group(lm, LIPS_OUTER),
            "left_brow":  _face_group(lm, LEFT_BROW),
            "right_brow": _face_group(lm, RIGHT_BROW),
        }

    sess = {
        "mode_key":      "face",
        "task_id":       session.task_id,
        "state":         session.state_name,
        "progress":      float(session.progress),
        "metric_value":  float(session.metric_value),
        "metric_target": float(session.metric_target),
    }
    if present:
        bs = face.blendshapes
        sess["yaw_norm"] = round(float(face.yaw_norm), 3)
        sess["pitch_norm"] = round(float(face.pitch_norm), 3)
        sess["roll_deg"] = round(float(face.roll_deg), 1)
        # Ham sinyaller (kalibrasyon/teşhis için UI'da gösterilir + loglanır)
        sess["debug"] = {
            "eyeBlinkL":  round(bs.get("eyeBlinkLeft", 0.0), 3),
            "eyeBlinkR":  round(bs.get("eyeBlinkRight", 0.0), 3),
            "smileL":     round(bs.get("mouthSmileLeft", 0.0), 3),
            "smileR":     round(bs.get("mouthSmileRight", 0.0), 3),
            "jawOpen":    round(bs.get("jawOpen", 0.0), 3),
            "browOuterL": round(bs.get("browOuterUpLeft", 0.0), 3),
            "browOuterR": round(bs.get("browOuterUpRight", 0.0), 3),
            "browInner":  round(bs.get("browInnerUp", 0.0), 3),
            "x_pitch":    round(float(face.pitch_norm), 3),
            "y_yaw":      round(float(face.yaw_norm), 3),
            "z_roll":     round(float(face.roll_deg), 1),
        }

    return {
        "type":         "state",
        "mode":         -1,            # el modlarından ayrı
        "mode_name":    "Yüz Görevi",
        "face_present": present,
        "face_groups":  groups,
        "session":      sess,
        "fps":          round(float(fps), 1),
    }


def serialize_bridge_shape(session, hands: list[HandResult], fps: float) -> dict[str, Any]:
    """BridgeShapeSession durumunu, frontend'in beklediği 'shape_eval'
    formatında serileştirir (mode 5 için GameManager yerine bunu kullanırız).

    tracer_state alanı INSTRUCTING/IDLE/POSITIONING/TRACING/VERIFIED/FAILED
    olduğu için mevcut frontend (shape_i18n.tracerInstruction) değişmeden çalışır.
    """
    st = session.state_name
    eval_state = "RESULT" if st in ("VERIFIED", "FAILED") else "RUNNING"
    tmpl = session.template
    sess = {
        "mode_key":          "shape_eval",
        "eval_mode":         "HUMAN TEST",
        "eval_state":        eval_state,
        "shape_label":       session.shape_label,
        "state_label":       st,
        "tracer_state":      st,
        "time_remaining":    float(session.time_remaining),
        "position_progress": float(session.position_progress),
        "draw_progress":     0.0,
        "instruct_remaining": 0.0,
        "point_count":       int(session.point_count),
        "min_trace_points":  int(tmpl.min_trace_points),
        "similarity":        float(session.similarity),
        "dtw_cost":          float(session.dtw_cost) if session.dtw_cost != float("inf") else 999.0,
        "result_time":       0.0,
        "template_path":     _round_path(session.template_waypoints),
        "user_path":         _round_path(session.traced_path),
        "start_point":       [round(tmpl.start_point[0], 4), round(tmpl.start_point[1], 4)],
        "end_point":         [round(tmpl.end_point[0], 4), round(tmpl.end_point[1], 4)],
        # Bridge'e özgü ek alanlar (robust akış HUD'u)
        "fist_progress":     float(session.fist_progress),
        "hand_lost":         bool(session.hand_lost),
    }
    return {
        "type": "state",
        "mode": 5,
        "mode_name": MODE_NAMES[5],
        "hands_present": len(hands) > 0,
        "hands": [_hand_to_json(h) for h in hands],
        "session": sess,
        "fps": round(float(fps), 1),
    }
