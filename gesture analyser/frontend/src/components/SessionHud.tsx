import type { StateMessage } from "../useGestureSocket";
import type { Task } from "../tasks";
import { translateCommand, STATE_TR } from "../liveness_i18n";
import { touchTitle, TOUCH_STATE_TR } from "../touch_i18n";
import { shapeName } from "../shape_i18n";
import { faceInstruction, faceMetricLabel, FACE_STATE_TR } from "../face_i18n";

interface Props {
  task: Task;
  state: StateMessage | null;
}

export function SessionHud({ task, state }: Props) {
  if (!state) {
    return (
      <div className="hud">
        <div className="hud-line dim">Bağlantı bekleniyor…</div>
      </div>
    );
  }

  const s = state.session ?? {};
  const lines: { label: string; value: string | number; tone?: string }[] = [];

  // ── Yüz görevi (el modlarından ayrı; hands alanı yok) ──────────────
  if (s.mode_key === "face") {
    lines.push({ label: "Mod", value: "Yüz Görevi" });
    lines.push({ label: "FPS", value: state.fps });
    lines.push({
      label: "Yüz",
      value: state.face_present ? "algılandı" : "yok",
      tone: state.face_present ? "ok" : "warn",
    });
    lines.push({ label: "Görev", value: faceInstruction(s.task_id) });
    lines.push({
      label: faceMetricLabel(s.task_id),
      value: `${Number(s.metric_value ?? 0).toFixed(2)} / ${Number(
        s.metric_target ?? 0,
      ).toFixed(2)}`,
    });
    lines.push({
      label: "İlerleme",
      value: `${Math.round((s.progress ?? 0) * 100)}%`,
    });
    lines.push({
      label: "Durum",
      value: FACE_STATE_TR[s.state] ?? s.state,
      tone: s.state === "SUCCESS" ? "ok" : undefined,
    });
    return (
      <div className="hud">
        <div className="hud-title">{task.title}</div>
        {lines.map((l) => (
          <div key={l.label} className={`hud-line${l.tone ? " hud-" + l.tone : ""}`}>
            <span className="hud-label">{l.label}</span>
            <span className="hud-value">{String(l.value)}</span>
          </div>
        ))}
      </div>
    );
  }

  // Genel bilgi (el modları)
  lines.push({ label: "Mod", value: state.mode_name });
  lines.push({ label: "FPS", value: state.fps });
  lines.push({
    label: "El",
    value: state.hands_present ? `${state.hands?.length ?? 0} algılandı` : "yok",
    tone: state.hands_present ? "ok" : "warn",
  });

  // Moda özgü alanlar
  if (s.mode_key === "gesture") {
    lines.push({ label: "Hedef", value: `${s.target} parmak` });
    lines.push({ label: "Algılanan", value: s.last_total ?? "—" });
    lines.push({ label: "Skor", value: s.score });
    lines.push({
      label: "Tutuş",
      value: `${Math.round((s.hold_progress ?? 0) * 100)}%`,
    });
    lines.push({
      label: "Durum",
      value: s.state,
      tone: s.state === "VALIDATED" ? "ok" : undefined,
    });
  } else if (s.mode_key === "math") {
    // equation_text, time_remaining, hold_progress, last_total — math_session.py
    lines.push({ label: "Algılanan", value: s.last_total ?? "—" });
    lines.push({
      label: "Tutuş",
      value: `${Math.round((s.hold_progress ?? 0) * 100)}%`,
    });
    lines.push({ label: "Skor", value: s.score ?? 0 });
    lines.push({
      label: "Kalan",
      value:
        s.time_remaining != null
          ? `${Math.max(0, Math.round(s.time_remaining))}s`
          : "—",
      tone:
        s.time_remaining != null && s.time_remaining < 10 ? "warn" : undefined,
    });
    lines.push({
      label: "Durum",
      value: s.state,
      tone:
        s.state === "SUCCESS"
          ? "ok"
          : s.state === "GAME_OVER"
          ? "warn"
          : undefined,
    });
  } else if (s.mode_key === "liveness") {
    // Komutu Türkçeleştir
    if (s.command_name)
      lines.push({ label: "Komut", value: translateCommand(s.command_name) });
    if (s.command_type)
      lines.push({ label: "Tip", value: s.command_type });
    lines.push({
      label: "İlerleme",
      value: `${s.challenges_completed ?? 0}/${s.num_challenges ?? 5}`,
    });
    if (s.verification_pct != null)
      lines.push({
        label: "Doğrulama",
        value: `%${Math.round(s.verification_pct)}`,
        tone: s.verification_pct >= 100 ? "ok" : undefined,
      });
    if (s.time_remaining != null) {
      const t = Math.max(0, Number(s.time_remaining));
      lines.push({
        label: "Kalan",
        value: `${t.toFixed(1)}s`,
        tone: t < 1.5 ? "warn" : undefined,
      });
    }
    if (s.is_wave_cmd && s.wave_reversals != null)
      lines.push({ label: "Sallama", value: `${s.wave_reversals} dönüş` });
    if (s.is_touch_cmd)
      lines.push({
        label: "Sıkıştırma",
        value: `${s.touch_frame_count}/10`,
      });
    if (s.area_change_pct != null)
      lines.push({
        label: "Derinlik Δ",
        value: `${s.area_change_pct >= 0 ? "+" : ""}${s.area_change_pct.toFixed(0)}%`,
      });
    lines.push({
      label: "Durum",
      value: STATE_TR[s.state] ?? s.state,
      tone:
        s.state === "SUCCESS" || s.state === "VERIFIED_100"
          ? "ok"
          : s.state === "FAILED"
          ? "warn"
          : undefined,
    });
    if (s.is_spoof_blocked)
      lines.push({ label: "Sahte", value: "engellendi", tone: "warn" });
  } else if (s.mode_key === "sequential") {
    if (s.current_step_text)
      lines.push({ label: "Adım", value: s.current_step_text });
    lines.push({
      label: "İlerleme",
      value: `${s.passed_count}/${s.total_steps}`,
    });
    lines.push({
      label: "Tutuş",
      value: `${Math.round((s.hold_progress ?? 0) * 100)}%`,
    });
    lines.push({ label: "Durum", value: s.state });
  } else if (s.mode_key === "touch_test") {
    if (s.command_name)
      lines.push({ label: "Komut", value: touchTitle(s.command_name) });
    lines.push({
      label: "İlerleme",
      value: `${s.passed_count}/${s.total ?? 5}`,
    });
    lines.push({
      label: "Tutuş",
      value: `${s.frame_count ?? 0}/${s.verify_frames ?? 10}`,
    });
    lines.push({
      label: "Durum",
      value: TOUCH_STATE_TR[s.state] ?? s.state,
      tone:
        s.state === "SUCCESS" || s.state === "COMPLETE" ? "ok" : undefined,
    });
  } else if (s.mode_key === "shape_eval") {
    if (s.shape_label)
      lines.push({ label: "Şekil", value: shapeName(s.shape_label) });
    if (s.tracer_state)
      lines.push({
        label: "Aşama",
        value: s.tracer_state,
        tone:
          s.tracer_state === "VERIFIED"
            ? "ok"
            : s.tracer_state === "FAILED"
            ? "warn"
            : undefined,
      });
    if (s.tracer_state === "TRACING") {
      lines.push({ label: "Nokta", value: `${s.point_count ?? 0}/${s.min_trace_points ?? 0}` });
      lines.push({
        label: "Kalan",
        value: `${Math.max(0, Number(s.time_remaining ?? 0)).toFixed(1)}s`,
      });
    }
    if (s.eval_state === "RESULT") {
      lines.push({
        label: "Benzerlik",
        value: `%${Math.round(s.similarity ?? 0)}`,
        tone: (s.similarity ?? 0) >= 70 ? "ok" : undefined,
      });
      lines.push({ label: "DTW", value: Number(s.dtw_cost ?? 0).toFixed(3) });
    }
  }

  return (
    <div className="hud">
      <div className="hud-title">{task.title}</div>
      {lines.map((l) => (
        <div key={l.label} className={`hud-line${l.tone ? " hud-" + l.tone : ""}`}>
          <span className="hud-label">{l.label}</span>
          <span className="hud-value">{String(l.value)}</span>
        </div>
      ))}
    </div>
  );
}
