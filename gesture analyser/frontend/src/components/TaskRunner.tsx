import { useEffect, useRef, useState } from "react";
import type { Task } from "../tasks";
import { useGestureSocket } from "../useGestureSocket";
import { TaskCanvas } from "./TaskCanvas";
import { SessionHud } from "./SessionHud";
import { translateCommand } from "../liveness_i18n";
import { touchTitle, touchHint } from "../touch_i18n";
import { tracerInstruction, shapeName } from "../shape_i18n";
import { seqStepName, SEQ_STATE_TR, SEQ_RESULT_TR } from "../sequential_i18n";
import { faceInstruction, faceMetricLabel, FACE_STATE_TR } from "../face_i18n";

const WS_URL =
  (window.location.protocol === "https:" ? "wss://" : "ws://") +
  window.location.host +
  "/ws";

const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;
// PythonProject1'in WaveDetector'ü 30fps için ayarlı (buffer_size=40,
// min_freq_hz=1.0). 15fps'de 40-örnek buffer ~2.7s dolduğu için normal
// el sallama (1-2Hz) min_freq filtresine takılıyordu. 30fps'de buffer
// ~1.3s dolar ve algılama doğru çalışır.
const TARGET_FPS = 30;
// JPEG kalitesi: orijinal masaüstü uygulaması ham frame veriyor; biz sıkıştırıp
// yolluyoruz. Düşük kalite (0.6) hareketli/bulanık eli bozup MediaPipe'ın hand
// landmark tespitini düşürüyordu (≈%47 boş frame, el sallama algılanmıyor).
// 0.92 sıkıştırma artefaktlarını azaltır → hareketli el çok daha güvenilir.
const JPEG_QUALITY = 0.92;
const LOG_TAG = "[gesture]";

interface Props {
  task: Task;
  onClose: () => void;
}

export function TaskRunner({ task, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [streamReady, setStreamReady] = useState(false);

  // Diagnostic counters
  const framesSentRef = useRef(0);
  const framesRecvRef = useRef(0);
  const lastLogTimeRef = useRef(0);

  // İstek-yanıt akış kontrolü: aynı anda yalnızca 1 frame "uçuşta".
  // Backend yanıtı gelmeden yeni frame göndermeyiz → kuyruk birikmez → gecikme olmaz.
  const inFlightRef = useRef(false);
  const lastSendTsRef = useRef(0);

  const { status, lastState, lastError, connect, send } = useGestureSocket(WS_URL);

  // Bağlantı durumu değişikliklerini logla
  useEffect(() => {
    console.log(LOG_TAG, "WS status →", status);
  }, [status]);

  // Sunucudan gelen her state'i debug için say + WAVE komutunda ayrıntılı logla
  useEffect(() => {
    if (!lastState) return;
    framesRecvRef.current += 1;
    // Yanıt geldi → bir sonraki frame gönderilebilir.
    inFlightRef.current = false;
    const s = lastState.session;
    if (s?.mode_key === "liveness" && s.command_type === "WAVE") {
      const h0 = lastState.hands?.[0];
      const entry = {
        t: Math.round(performance.now()),
        state: s.state,
        time_rem: Number(s.time_remaining ?? 0).toFixed(2),
        reversals: s.wave_reversals,
        buffer: `${s.wave_buffer_len ?? "?"}/${s.wave_buffer_max ?? "?"}`,
        total_disp: Number(s.wave_total_disp ?? 0).toFixed(3),
        min_total: s.wave_min_total,
        span_s: Number(s.wave_span_s ?? 0).toFixed(2),
        // El takip teşhisi: kaç el, hands[0] hangi el, bileğin x'i
        n_hands: lastState.hands?.length ?? 0,
        hand0: h0?.handedness ?? "—",
        wrist_x: h0 ? Number(h0.landmarks[0].x).toFixed(3) : null,
        fps: lastState.fps,
      };
      // Okunabilir tek-satır string (read_console_messages nesneyi açmıyor)
      console.log(
        LOG_TAG,
        `WAVE st=${entry.state} hand0=${entry.hand0} nH=${entry.n_hands} ` +
          `wx=${entry.wrist_x} rev=${entry.reversals} disp=${entry.total_disp}/` +
          `${entry.min_total} buf=${entry.buffer} span=${entry.span_s}s t=${entry.time_rem}`,
      );
      // javascript_exec ile sonradan analiz için global ring buffer
      const w = window as any;
      if (!w.__waveLog) w.__waveLog = [];
      w.__waveLog.push(entry);
      if (w.__waveLog.length > 400) w.__waveLog.shift();
    }

    // Shape Eval teşhisi: her frame tracer durumu + nokta + el sayısı
    if (s?.mode_key === "shape_eval") {
      const h0 = lastState.hands?.[0];
      const tip = h0?.landmarks?.[8]; // INDEX_TIP
      const sp = s.start_point;
      const ep = s.end_point;
      const dist = (a: any, p: any) =>
        a && p ? Math.hypot(a.x - p[0], a.y - p[1]) : null;
      const dStart = tip && sp ? dist(tip, sp) : null;
      const dEnd = tip && ep ? dist(tip, ep) : null;
      const entry = {
        t: Math.round(performance.now()),
        tracer: s.tracer_state,
        eval: s.eval_state,
        points: s.point_count ?? 0,
        min_pts: s.min_trace_points ?? 0,
        time_rem: Number(s.time_remaining ?? 0).toFixed(2),
        n_hands: lastState.hands?.length ?? 0,
        hand0: h0?.handedness ?? "—",
        ix: tip ? Number(tip.x).toFixed(3) : null,
        iy: tip ? Number(tip.y).toFixed(3) : null,
        d_start: dStart != null ? Number(dStart).toFixed(3) : null,
        d_end: dEnd != null ? Number(dEnd).toFixed(3) : null,
        similarity: s.similarity,
        dtw: s.dtw_cost,
        fps: lastState.fps,
      };
      const w = window as any;
      if (!w.__shapeLog) w.__shapeLog = [];
      const prev = w.__shapeLog[w.__shapeLog.length - 1];
      // Sadece durum/nokta değişince logla (gürültüyü azalt) + her zaman buffer'a ekle
      if (
        !prev ||
        prev.tracer !== entry.tracer ||
        prev.eval !== entry.eval ||
        prev.n_hands !== entry.n_hands
      ) {
        console.log(
          LOG_TAG,
          `SHAPE tracer=${entry.tracer} eval=${entry.eval} pts=${entry.points}/${entry.min_pts} ` +
            `nH=${entry.n_hands} t=${entry.time_rem} sim=${entry.similarity} dtw=${entry.dtw}`,
        );
      }
      w.__shapeLog.push(entry);
      if (w.__shapeLog.length > 600) w.__shapeLog.shift();

      // ── Kalıcı tur özetleri (idle frame'ler bunu silemez) ──────────────
      // TRACING sırasında yolu biriktir; tamamlanınca DTW/benzerlik + yol
      // özetini __shapeRounds'a yaz. Böylece reddedilen şeklin verisi korunur.
      if (!w.__shapeRounds) w.__shapeRounds = [];
      const prevTracer = w.__prevTracer ?? null;
      const cur = entry.tracer;
      if (cur === "TRACING") {
        if (prevTracer !== "TRACING") w.__curPath = []; // yeni çizim
        if (entry.ix != null && entry.iy != null)
          w.__curPath.push([parseFloat(entry.ix), parseFloat(entry.iy)]);
      }
      if (
        (cur === "VERIFIED" || cur === "FAILED") &&
        prevTracer === "TRACING"
      ) {
        const path = (w.__curPath ?? []) as number[][];
        // outlier / sıçrama analizi
        let maxStep = 0,
          bigSteps = 0;
        for (let k = 1; k < path.length; k++) {
          const d = Math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]);
          if (d > maxStep) maxStep = d;
          if (d > 0.12) bigSteps++;
        }
        const xs = path.map((p) => p[0]),
          ys = path.map((p) => p[1]);
        const cx = xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
        const cy = ys.reduce((a, b) => a + b, 0) / (ys.length || 1);
        const radii = path.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
        const maxR = radii.length ? Math.max(...radii) : 0;
        const meanR = radii.length ? radii.reduce((a, b) => a + b, 0) / radii.length : 0;
        w.__shapeRounds.push({
          result: cur,
          dtw: entry.dtw,
          sim: entry.similarity,
          points: path.length,
          min_pts: entry.min_pts,
          shape: s.shape_label,
          bbox: xs.length
            ? `x[${Math.min(...xs).toFixed(2)}-${Math.max(...xs).toFixed(2)}] y[${Math.min(...ys).toFixed(2)}-${Math.max(...ys).toFixed(2)}]`
            : "—",
          maxR_meanR: meanR ? +(maxR / meanR).toFixed(2) : 0,
          maxStep: +maxStep.toFixed(3),
          bigSteps,
        });
        if (w.__shapeRounds.length > 30) w.__shapeRounds.shift();
        console.log(LOG_TAG, "SHAPE ROUND", JSON.stringify(w.__shapeRounds[w.__shapeRounds.length - 1]));
      }
      w.__prevTracer = cur;
    }

    // ── Yüz görevi teşhisi: ham sinyalleri logla (kalibrasyon) ──────────
    if (s?.mode_key === "face") {
      const d = s.debug ?? {};
      const entry = {
        t: Math.round(performance.now()),
        task: s.task_id,
        state: s.state,
        progress: Number(s.progress ?? 0).toFixed(2),
        metric: Number(s.metric_value ?? 0).toFixed(3),
        target: Number(s.metric_target ?? 0).toFixed(3),
        face: lastState.face_present,
        ...d, // eyeBlinkL/R, smileL/R, jawOpen, browOuterL/R, browInner, x_pitch, y_yaw, z_roll
      };
      const w = window as any;
      if (!w.__faceLog) w.__faceLog = [];
      w.__faceLog.push(entry);
      if (w.__faceLog.length > 500) w.__faceLog.shift();
      // Konsola kompakt: eksenler + aktif metrik
      console.log(
        LOG_TAG,
        `FACE ${entry.task} st=${entry.state} prog=${entry.progress} ` +
          `m=${entry.metric}/${entry.target} | X(pitch)=${d.x_pitch} Y(yaw)=${d.y_yaw} Z(roll)=${d.z_roll} | ` +
          `blinkL=${d.eyeBlinkL} blinkR=${d.eyeBlinkR} smL=${d.smileL} smR=${d.smileR} ` +
          `jaw=${d.jawOpen} browL=${d.browOuterL} browR=${d.browOuterR} browIn=${d.browInner}`,
      );
    }
  }, [lastState]);

  // Her saniye throughput logla (frames sent / recv)
  useEffect(() => {
    const id = setInterval(() => {
      const now = performance.now();
      const dt = lastLogTimeRef.current === 0 ? 1000 : now - lastLogTimeRef.current;
      console.log(
        LOG_TAG,
        "throughput",
        `sent=${framesSentRef.current} recv=${framesRecvRef.current} dt=${Math.round(dt)}ms`,
      );
      framesSentRef.current = 0;
      framesRecvRef.current = 0;
      lastLogTimeRef.current = now;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Webcam başlat
  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: FRAME_WIDTH, height: FRAME_HEIGHT, facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream?.getTracks().forEach((t) => t.stop());
          return;
        }
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          // streamReady'yi kırılgan play() promise'ine bağlamıyoruz: bazı
          // tarayıcılarda play() promise'i hiç çözülmeyebilir (srcObject
          // değişimi/yeniden yükleme yarışı). Bunun yerine video gerçekten
          // veri aldığında ('loadeddata' veya readyState>=2) hazır sayıyoruz.
          const markReady = () => {
            if (!cancelled) {
              console.log(LOG_TAG, "kamera hazır (loadeddata)");
              setStreamReady(true);
            }
          };
          if (v.readyState >= 2) markReady();
          else v.addEventListener("loadeddata", markReady, { once: true });
          // play()'i tetikle ama promise'ini bekleme; hatası kritik değil.
          v.play().catch((err) =>
            console.warn(LOG_TAG, "video.play() reddetti (yoksayıldı):", err?.message),
          );
        }
      } catch (e: any) {
        console.error(LOG_TAG, "getUserMedia hatası:", e?.name, e?.message);
        setCameraError(e?.message ?? "Kameraya erişilemedi.");
      }
    })();

    return () => {
      cancelled = true;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // WS bağlan
  useEffect(() => {
    connect();
  }, [connect]);

  // Mod ayarla — yüz görevi ise set_face_task, el görevi ise set_mode.
  useEffect(() => {
    if (status !== "open") return;
    if (task.family === "face") {
      send({ type: "set_face_task", task: task.faceTaskId });
    } else {
      send({
        type: "set_mode",
        mode: task.modeIndex ?? 0,
        prefer_cmd: task.preferredCommand ?? null,
      });
    }
  }, [
    status,
    task.family,
    task.faceTaskId,
    task.modeIndex,
    task.preferredCommand,
    send,
  ]);

  // Frame gönderim döngüsü — İSTEK-YANIT (adaptif) modeli.
  // setInterval ile sürekli kontrol eder ama yalnızca bir önceki frame'in
  // yanıtı geldiyse (inFlightRef=false) yeni frame yollar. inFlightRef guard'ı
  // zamanlayıcıdan bağımsız çalıştığı için kuyruk birikmesi (backpressure)
  // imkânsız; gönderim hızı backend'in işleme hızına otomatik uyar.
  //
  // requestAnimationFrame yerine setInterval kullanıyoruz çünkü rAF sekme arka
  // plana alındığında tamamen durur (bu da frame akışını kesiyordu). setInterval
  // arka planda 1sn'e throttle edilse de bağlantıyı canlı tutar; sekme önde
  // olduğunda tam hızda (≈30fps) çalışır.
  useEffect(() => {
    if (!streamReady) return;
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
      captureCanvasRef.current.width = FRAME_WIDTH;
      captureCanvasRef.current.height = FRAME_HEIGHT;
    }
    const cv = captureCanvasRef.current!;
    const ctx = cv.getContext("2d")!;

    const tick = () => {
      const v = videoRef.current;
      const now = performance.now();

      // Güvenlik: yanıt 1sn'den uzun gelmediyse kilidi aç (paket kaybı vs.).
      if (inFlightRef.current && now - lastSendTsRef.current > 1000) {
        console.warn(LOG_TAG, "yanıt zaman aşımı — frame kilidi açılıyor");
        inFlightRef.current = false;
      }

      if (
        v &&
        v.readyState >= 2 &&
        status === "open" &&
        !inFlightRef.current
      ) {
        ctx.drawImage(v, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        const data = cv.toDataURL("image/jpeg", JPEG_QUALITY);
        if (send({ type: "frame", data })) {
          inFlightRef.current = true;
          lastSendTsRef.current = now;
          framesSentRef.current += 1;
        }
      }
    };

    const id = setInterval(tick, Math.round(1000 / TARGET_FPS));
    return () => clearInterval(id);
  }, [streamReady, status, send]);

  return (
    <div className="runner">
      <header className="runner-head">
        <button className="back" onClick={onClose}>
          ← Görevlere dön
        </button>
        <h2>
          {task.icon} {task.title}
        </h2>
        <button
          className="restart"
          onClick={() => send({ type: "restart" })}
          disabled={status !== "open"}
        >
          ↻ Yeniden başlat
        </button>
      </header>

      <div className="runner-body">
        <div className="runner-stage">
          <video
            ref={videoRef}
            muted
            playsInline
            style={{ display: "none" }}
          />
          <TaskCanvas
            videoRef={videoRef}
            hands={lastState?.hands ?? []}
            width={FRAME_WIDTH}
            height={FRAME_HEIGHT}
            face={
              lastState?.session?.mode_key === "face"
                ? lastState.face_groups ?? null
                : null
            }
            shape={
              lastState?.session?.mode_key === "shape_eval"
                ? {
                    templatePath: lastState.session.template_path ?? [],
                    userPath: lastState.session.user_path ?? [],
                    startPoint: lastState.session.start_point ?? null,
                    endPoint: lastState.session.end_point ?? null,
                    tracerState: lastState.session.tracer_state ?? null,
                  }
                : null
            }
          />

          {/* Matematik modu için denklem overlay'i */}
          {lastState?.session?.mode_key === "math" &&
            lastState.session.state !== "GAME_OVER" && (
              <div className="math-overlay">
                <div className="math-eq">
                  {lastState.session.equation_text ?? "…"}
                </div>
                <div className="math-meta">
                  <span>
                    ⏱ {Math.max(0, Math.round(lastState.session.time_remaining ?? 0))}s
                  </span>
                  <span>Skor: {lastState.session.score ?? 0}</span>
                  {lastState.session.last_total != null && (
                    <span>Algılanan: {lastState.session.last_total}</span>
                  )}
                </div>
                {(lastState.session.hold_progress ?? 0) > 0 && (
                  <div className="hold-bar">
                    <div
                      className="hold-bar-fill"
                      style={{
                        width: `${Math.round(
                          (lastState.session.hold_progress ?? 0) * 100,
                        )}%`,
                      }}
                    />
                  </div>
                )}
              </div>
            )}

          {lastState?.session?.mode_key === "math" &&
            lastState.session.state === "GAME_OVER" && (
              <div className="math-gameover">
                <div className="big">SÜRE DOLDU</div>
                <div>Son skor: {lastState.session.score ?? 0}</div>
                <button
                  className="restart"
                  onClick={() => send({ type: "restart" })}
                >
                  ↻ Yeniden başlat
                </button>
              </div>
            )}

          {/* Parmak Say (Normal/gesture) overlay'i */}
          {lastState?.session?.mode_key === "gesture" && (
            <div className="live-overlay">
              <div className="live-progress-pill">
                Skor: {lastState.session.score ?? 0}
              </div>
              <div
                className={
                  "live-cmd " +
                  (lastState.session.state === "VALIDATED"
                    ? "live-cmd-ok"
                    : "")
                }
              >
                {lastState.session.state === "VALIDATED"
                  ? "✓ DOĞRULANDI"
                  : `${lastState.session.target ?? "?"} parmak göster`}
              </div>
              {lastState.session.state !== "VALIDATED" && (
                <>
                  <div className="live-timer-row">
                    <span>
                      Algılanan: {lastState.session.last_total ?? "—"}
                    </span>
                    <span>
                      Tutuş:{" "}
                      {Math.round(
                        (lastState.session.hold_progress ?? 0) * 100,
                      )}
                      %
                    </span>
                  </div>
                  {(lastState.session.hold_progress ?? 0) > 0 && (
                    <div className="hold-bar">
                      <div
                        className="hold-bar-fill"
                        style={{
                          width: `${Math.round(
                            (lastState.session.hold_progress ?? 0) * 100,
                          )}%`,
                        }}
                      />
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Sıralı Görev overlay'i */}
          {lastState?.session?.mode_key === "sequential" &&
            lastState.session.state !== "COMPLETE" && (
              <div className="live-overlay">
                <div className="live-progress-pill">
                  Adım {(lastState.session.current_idx ?? 0) + 1}/
                  {lastState.session.total_steps ?? 19}
                  &nbsp;·&nbsp;{lastState.session.passed_count ?? 0} geçti
                </div>

                <div
                  className={
                    "live-cmd " +
                    (lastState.session.state === "STEP_DONE"
                      ? "live-cmd-ok"
                      : lastState.session.state === "STEP_TIMEOUT"
                      ? "live-cmd-fail"
                      : "")
                  }
                >
                  {lastState.session.state === "STEP_DONE"
                    ? "✓ GEÇTİ"
                    : lastState.session.state === "STEP_TIMEOUT"
                    ? "✗ SÜRE DOLDU"
                    : seqStepName(lastState.session.step_name)}
                </div>

                {(lastState.session.state === "ACTIVE" ||
                  lastState.session.state === "HOLDING") && (
                  <>
                    <div className="live-timer-row">
                      <span>
                        ⏱{" "}
                        {Math.max(
                          0,
                          Number(lastState.session.time_remaining ?? 0),
                        ).toFixed(1)}
                        s
                      </span>
                      <span>
                        {SEQ_STATE_TR[lastState.session.state] ??
                          lastState.session.state}
                      </span>
                    </div>
                    {(lastState.session.hold_progress ?? 0) > 0 && (
                      <div className="hold-bar">
                        <div
                          className="hold-bar-fill"
                          style={{
                            width: `${Math.round(
                              (lastState.session.hold_progress ?? 0) * 100,
                            )}%`,
                          }}
                        />
                      </div>
                    )}
                  </>
                )}

                {/* Adım checklist'i (kompakt) */}
                <div className="seq-grid">
                  {(lastState.session.results ?? []).map(
                    (r: string, i: number) => (
                      <span
                        key={i}
                        className={
                          "seq-chip" +
                          (r === "PASSED"
                            ? " seq-chip-ok"
                            : r === "TIMED_OUT"
                            ? " seq-chip-fail"
                            : "") +
                          (i === lastState.session.current_idx
                            ? " seq-chip-active"
                            : "")
                        }
                        title={`Adım ${i + 1}`}
                      >
                        {SEQ_RESULT_TR[r] ?? "•"}
                      </span>
                    ),
                  )}
                </div>
              </div>
            )}

          {/* Sıralı Görev tamamlanma ekranı */}
          {lastState?.session?.mode_key === "sequential" &&
            lastState.session.state === "COMPLETE" && (
              <div className="math-gameover">
                <div className="big" style={{ color: "var(--green)" }}>
                  TÜM ADIMLAR TAMAM ✓
                </div>
                <div>
                  {lastState.session.passed_count ?? 0}/
                  {lastState.session.total_steps ?? 19} adım geçildi
                </div>
                <button
                  className="restart"
                  onClick={() => send({ type: "restart" })}
                >
                  ↻ Yeniden başlat
                </button>
              </div>
            )}

          {/* Liveness modu overlay'i */}
          {lastState?.session?.mode_key === "liveness" &&
            lastState.session.state !== "VERIFIED_100" && (
              <div className="live-overlay">
                <div className="live-progress-pill">
                  Görev {(lastState.session.challenges_completed ?? 0) + 1}/
                  {lastState.session.num_challenges ?? 5}
                  &nbsp;·&nbsp;%{Math.round(lastState.session.verification_pct ?? 0)}
                </div>

                {/* Arming aşaması — HAND_FLIP / PEEK_A_BOO yanlış-pozitif önleme */}
                {lastState.session.arming && (
                  <>
                    <div className="live-cmd" style={{ fontSize: 22 }}>
                      ⌛ Hazırlanıyor — {lastState.session.arming_text}
                    </div>
                    <div className="hold-bar">
                      <div
                        className="hold-bar-fill"
                        style={{
                          width: `${Math.round(
                            (lastState.session.arming_progress ?? 0) * 100,
                          )}%`,
                        }}
                      />
                    </div>
                  </>
                )}

                {!lastState.session.arming && (
                  <div
                    className={
                      "live-cmd " +
                      (lastState.session.state === "SUCCESS"
                        ? "live-cmd-ok"
                        : lastState.session.state === "FAILED"
                        ? "live-cmd-fail"
                        : "")
                    }
                  >
                    {lastState.session.state === "SUCCESS"
                      ? "✓ DOĞRULANDI"
                      : lastState.session.state === "FAILED"
                      ? lastState.session.is_spoof_blocked
                        ? "⚠ SAHTECİLİK TESPİT EDİLDİ"
                        : "✗ SÜRE DOLDU"
                      : translateCommand(lastState.session.command_name) || "…"}
                  </div>
                )}

                {!lastState.session.arming && lastState.session.state === "ACTIVE" && (
                  <>
                    <div className="live-timer-row">
                      <span>
                        ⏱ {Math.max(0, Number(lastState.session.time_remaining ?? 0)).toFixed(1)}s
                      </span>
                      {lastState.session.is_wave_cmd && (
                        <span>👋 {lastState.session.wave_reversals ?? 0} dönüş</span>
                      )}
                      {lastState.session.area_change_pct != null && (
                        <span>
                          📏{" "}
                          {lastState.session.area_change_pct >= 0 ? "+" : ""}
                          {Math.round(lastState.session.area_change_pct)}%
                        </span>
                      )}
                    </div>

                    {(lastState.session.is_touch_cmd ||
                      (lastState.session.touch_frame_progress ?? 0) > 0) && (
                      <div className="hold-bar">
                        <div
                          className="hold-bar-fill"
                          style={{
                            width: `${Math.round(
                              (lastState.session.touch_frame_progress ?? 0) *
                                100,
                            )}%`,
                          }}
                        />
                      </div>
                    )}

                    {(lastState.session.spoof_warning_progress ?? 0) > 0 && (
                      <div className="spoof-bar">
                        <div
                          className="spoof-bar-fill"
                          style={{
                            width: `${Math.round(
                              (lastState.session.spoof_warning_progress ?? 0) *
                                100,
                            )}%`,
                          }}
                        />
                      </div>
                    )}
                  </>
                )}

                {lastState.session.state === "DEBOUNCE" && (
                  <div className="hold-bar">
                    <div
                      className="hold-bar-fill"
                      style={{
                        width: `${Math.round(
                          (lastState.session.debounce_progress ?? 0) * 100,
                        )}%`,
                      }}
                    />
                  </div>
                )}
              </div>
            )}

          {lastState?.session?.mode_key === "liveness" &&
            lastState.session.state === "VERIFIED_100" && (
              <div className="math-gameover">
                <div className="big" style={{ color: "var(--green)" }}>
                  ERİŞİM ONAYLANDI
                </div>
                <div>Tüm canlılık görevleri tamamlandı (%100)</div>
                <button
                  className="restart"
                  onClick={() => send({ type: "restart" })}
                >
                  ↻ Yeniden başlat
                </button>
              </div>
            )}

          {/* Touch Test (Sıkıştır) overlay'i */}
          {lastState?.session?.mode_key === "touch_test" &&
            lastState.session.state !== "COMPLETE" && (
              <div className="live-overlay">
                <div className="live-progress-pill">
                  {`Komut ${(lastState.session.current_idx ?? 0) + 1}/${
                    lastState.session.total ?? 5
                  } · ${lastState.session.passed_count ?? 0} geçti`}
                </div>

                <div
                  className={
                    "live-cmd " +
                    (lastState.session.state === "SUCCESS" ? "live-cmd-ok" : "")
                  }
                >
                  {lastState.session.state === "SUCCESS"
                    ? "✓ DOĞRULANDI"
                    : "🤏 " + touchTitle(lastState.session.command_name)}
                </div>

                {lastState.session.state === "ACTIVE" && (
                  <>
                    <div className="live-timer-row">
                      <span>{touchHint(lastState.session.command_name)}</span>
                      <span>
                        {lastState.session.frame_count ?? 0}/
                        {lastState.session.verify_frames ?? 10}
                      </span>
                    </div>
                    <div className="hold-bar">
                      <div
                        className="hold-bar-fill"
                        style={{
                          width: `${Math.round(
                            (lastState.session.hold_progress ?? 0) * 100,
                          )}%`,
                        }}
                      />
                    </div>
                  </>
                )}

                {/* 5'li checklist */}
                <div className="touch-checklist">
                  {(lastState.session.commands ?? []).map(
                    (c: any, i: number) => (
                      <span
                        key={c.name}
                        className={
                          "touch-chip" +
                          (c.passed ? " touch-chip-ok" : "") +
                          (i === lastState.session.current_idx
                            ? " touch-chip-active"
                            : "")
                        }
                      >
                        {c.passed ? "✓" : i + 1} {touchTitle(c.name)}
                      </span>
                    ),
                  )}
                </div>
              </div>
            )}

          {lastState?.session?.mode_key === "touch_test" &&
            lastState.session.state === "COMPLETE" && (
              <div className="math-gameover">
                <div className="big" style={{ color: "var(--green)" }}>
                  TÜM SIKIŞTIRMALAR TAMAM ✓
                </div>
                <div>5/5 parmak-değdirme kombinasyonu doğrulandı</div>
                <button
                  className="restart"
                  onClick={() => send({ type: "restart" })}
                >
                  ↻ Yeniden başlat
                </button>
              </div>
            )}

          {/* Shape Eval (Şekil Çiz) overlay'i */}
          {lastState?.session?.mode_key === "shape_eval" &&
            (() => {
              const ss = lastState.session;
              const instr = tracerInstruction(ss.tracer_state, ss.shape_label);
              return (
                <div className="shape-ov">
                  {/* Kompakt üst bar — şekli örtmesin */}
                  <div
                    className={
                      "shape-instr" +
                      (instr.tone === "ok"
                        ? " shape-instr-ok"
                        : instr.tone === "fail"
                        ? " shape-instr-fail"
                        : instr.tone === "go"
                        ? " shape-instr-go"
                        : "")
                    }
                  >
                    <span className="shape-badge">{shapeName(ss.shape_label)}</span>
                    <span className="shape-instr-text">{instr.text}</span>
                    {ss.tracer_state === "TRACING" && (
                      <span className="shape-meta">
                        ⏱{Math.max(0, Number(ss.time_remaining ?? 0)).toFixed(1)}s ·{" "}
                        {ss.point_count ?? 0}/{ss.min_trace_points ?? 0}
                        {ss.hand_lost ? " · ✋ el bekleniyor" : ""}
                      </span>
                    )}
                    {ss.eval_state === "RESULT" && (
                      <span className="shape-meta">
                        %{Math.round(ss.similarity ?? 0)} · DTW{" "}
                        {Number(ss.dtw_cost ?? 0).toFixed(3)}
                      </span>
                    )}
                  </div>

                  {/* POSITIONING geri sayım çubuğu — üst barın hemen altında */}
                  {ss.tracer_state === "POSITIONING" && (
                    <div className="shape-holdbar">
                      <div
                        className="hold-bar-fill"
                        style={{
                          width: `${Math.round((ss.position_progress ?? 0) * 100)}%`,
                        }}
                      />
                    </div>
                  )}

                  {/* Yumruk ile bitirme ilerlemesi (titremeye dayanıklı) */}
                  {ss.tracer_state === "TRACING" && (ss.fist_progress ?? 0) > 0 && (
                    <div className="shape-holdbar">
                      <div
                        className="hold-bar-fill"
                        style={{
                          width: `${Math.round((ss.fist_progress ?? 0) * 100)}%`,
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })()}

          {/* Yüz görevi overlay'i */}
          {lastState?.session?.mode_key === "face" &&
            (() => {
              const fs = lastState.session;
              const st = fs.state as string;
              const isSuccess = st === "SUCCESS";
              const isNoFace = st === "NO_FACE";
              const isArming = st === "ARMING";
              return (
                <div className="live-overlay">
                  <div className="live-progress-pill count-pill-face">
                    Yüz Görevi
                  </div>

                  <div
                    className={
                      "live-cmd " + (isSuccess ? "live-cmd-ok" : "")
                    }
                    style={{ fontSize: 24 }}
                  >
                    {isSuccess
                      ? "✓ DOĞRULANDI"
                      : isNoFace
                      ? "🙂 Yüzünü kameraya göster"
                      : isArming
                      ? "⌛ Hazırlanıyor — yüzünü düz tut"
                      : faceInstruction(fs.task_id)}
                  </div>

                  {!isSuccess && !isNoFace && (
                    <>
                      <div className="live-timer-row">
                        <span>{faceMetricLabel(fs.task_id)}</span>
                        <span>
                          {Number(fs.metric_value ?? 0).toFixed(2)} /{" "}
                          {Number(fs.metric_target ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <div className="hold-bar">
                        <div
                          className="hold-bar-fill"
                          style={{
                            width: `${Math.round((fs.progress ?? 0) * 100)}%`,
                          }}
                        />
                      </div>
                    </>
                  )}

                  {/* Canlı eksen + blendshape teşhis paneli */}
                  {fs.debug && (
                    <div className="face-debug">
                      <div className="face-debug-row">
                        <span>X eksen (pitch): <b>{fs.debug.x_pitch}</b></span>
                        <span>Y eksen (yaw): <b>{fs.debug.y_yaw}</b></span>
                        <span>Z eksen (roll): <b>{fs.debug.z_roll}°</b></span>
                      </div>
                      <div className="face-debug-row">
                        <span>göz L/R: <b>{fs.debug.eyeBlinkL}</b>/<b>{fs.debug.eyeBlinkR}</b></span>
                        <span>gülüş L/R: <b>{fs.debug.smileL}</b>/<b>{fs.debug.smileR}</b></span>
                        <span>ağız: <b>{fs.debug.jawOpen}</b></span>
                      </div>
                      <div className="face-debug-row">
                        <span>kaş dış L/R: <b>{fs.debug.browOuterL}</b>/<b>{fs.debug.browOuterR}</b></span>
                        <span>kaş iç: <b>{fs.debug.browInner}</b></span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

          {cameraError && (
            <div className="overlay-error">
              Kamera hatası: {cameraError}
            </div>
          )}
          {!streamReady && !cameraError && (
            <div className="overlay-info">Kamera başlatılıyor…</div>
          )}
        </div>

        <aside className="runner-aside">
          <div className="conn-line">
            Bağlantı:&nbsp;
            <span className={`dot dot-${status}`}></span>
            {status}
          </div>
          {lastError && <div className="err-line">⚠ {lastError}</div>}
          <SessionHud task={task} state={lastState} />
          {task.family === "face" ? (
            <p className="hint">
              Bu kart <strong>{task.title}</strong> için bridge tarafındaki{" "}
              <code>FaceTracker</code> + <code>FaceTaskSession</code> (MediaPipe
              FaceLandmarker) akışını çalıştırır. PythonProject1'e dokunulmadan
              eklenmiştir.
            </p>
          ) : (
            <p className="hint">
              Bu kart <strong>{task.title}</strong> için Gesture Analyser
              motoru modu&nbsp;
              <code>{task.modeIndex}</code>'a bağlandı. Komutlar ve cevaplar
              backend'deki <code>GameManager</code> tarafından gerçek-zamanlı
              işleniyor.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
