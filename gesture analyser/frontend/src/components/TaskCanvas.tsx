import { useEffect, useRef } from "react";
import type { Hand, FaceGroups } from "../useGestureSocket";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const HAND_COLOR: Record<string, { line: string; dot: string; box: string }> = {
  Left: { line: "#ffb400", dot: "#ff6400", box: "#ff3030" },
  Right: { line: "#b400b4", dot: "#a070ff", box: "#3070ff" },
};

export interface ShapeOverlay {
  templatePath: number[][];
  userPath: number[][];
  startPoint: number[] | null;
  endPoint: number[] | null;
  tracerState: string | null;
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  hands: Hand[];
  width: number;
  height: number;
  shape?: ShapeOverlay | null;
  face?: FaceGroups | null;
}

export function TaskCanvas({ videoRef, hands, width, height, shape, face }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    const video = videoRef.current;
    if (!cv || !video) return;

    const ctx = cv.getContext("2d");
    if (!ctx) return;

    cv.width = width;
    cv.height = height;
    ctx.clearRect(0, 0, width, height);

    // Webcam frame'i mirror'lanmış olarak çiziyoruz (Python tarafı flip yapıyor).
    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    try {
      ctx.drawImage(video, 0, 0, width, height);
    } catch {
      /* video henüz hazır değilse görmezden gel */
    }
    ctx.restore();

    // ── Yüz mesh overlay'i (Yüz Görevleri) ──────────────────────────────
    if (face) {
      const drawGroup = (
        pts: number[][] | undefined,
        color: string,
        closed: boolean,
      ) => {
        if (!pts || pts.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach(([x, y], i) => {
          const px = x * width;
          const py = y * height;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        if (closed) ctx.closePath();
        ctx.stroke();
      };
      drawGroup(face.oval, "rgba(160, 130, 255, 0.7)", true);
      drawGroup(face.left_eye, "#37e0ff", true);
      drawGroup(face.right_eye, "#37e0ff", true);
      drawGroup(face.lips, "#ff7eb3", true);
      drawGroup(face.left_brow, "#3ddc97", false);
      drawGroup(face.right_brow, "#3ddc97", false);
    }

    // ── Şekil overlay'i (Shape Eval) — landmark'larla aynı koord. uzayı ──
    if (shape) {
      // Şablon yolu (yeşil yarı saydam rehber)
      if (shape.templatePath && shape.templatePath.length > 1) {
        ctx.strokeStyle = "rgba(80, 220, 140, 0.55)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        shape.templatePath.forEach(([x, y], i) => {
          const px = x * width;
          const py = y * height;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Kullanıcının çizdiği yol (turuncu/kırmızı, dolu)
      if (shape.userPath && shape.userPath.length > 1) {
        ctx.strokeStyle = "#ff5b5b";
        ctx.lineWidth = 3;
        ctx.lineJoin = "round";
        ctx.beginPath();
        shape.userPath.forEach(([x, y], i) => {
          const px = x * width;
          const py = y * height;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Başla noktası (cyan halka + etiket) — trigger yarıçapı 0.06
      if (shape.startPoint) {
        const sx = shape.startPoint[0] * width;
        const sy = shape.startPoint[1] * height;
        const r = 0.06 * width;
        ctx.strokeStyle = "#37e0ff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#37e0ff";
        ctx.font = "bold 13px system-ui, sans-serif";
        ctx.fillText("BAŞLA", sx - 20, sy - r - 6);
      }

      // Bitir noktası — yalnızca ekran içi ([0,1]) ve start'tan farklıysa çiz.
      // (Bridge bitiş-noktası tetiğini kapatmak için end_point'i ekran dışına
      //  (99,99) taşıyabilir; o durumda BİTİR çizilmez.)
      const ep = shape.endPoint;
      const epOnScreen =
        ep && ep[0] >= 0 && ep[0] <= 1 && ep[1] >= 0 && ep[1] <= 1;
      if (
        ep &&
        epOnScreen &&
        shape.startPoint &&
        (ep[0] !== shape.startPoint[0] || ep[1] !== shape.startPoint[1])
      ) {
        const ex = ep[0] * width;
        const ey = ep[1] * height;
        ctx.fillStyle = "#ffcd3c";
        ctx.beginPath();
        ctx.arc(ex, ey, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = "bold 13px system-ui, sans-serif";
        ctx.fillText("BİTİR", ex - 18, ey + 24);
      }
    }

    // Landmark'lar zaten mirror sonrası koordinatlarda — direkt çizebiliriz.
    for (const hand of hands) {
      const color = HAND_COLOR[hand.handedness] ?? HAND_COLOR.Right;

      ctx.strokeStyle = color.line;
      ctx.lineWidth = 2;
      for (const [s, e] of HAND_CONNECTIONS) {
        const a = hand.landmarks[s];
        const b = hand.landmarks[e];
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x * width, a.y * height);
        ctx.lineTo(b.x * width, b.y * height);
        ctx.stroke();
      }

      ctx.fillStyle = color.dot;
      for (const lm of hand.landmarks) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // Etiket
      const w = hand.landmarks[0];
      if (w) {
        ctx.fillStyle = color.box;
        ctx.font = "bold 14px system-ui, sans-serif";
        ctx.fillText(hand.handedness, w.x * width - 18, w.y * height + 22);
      }

      // Bounding kutu
      const xs = hand.landmarks.map((l) => l.x * width);
      const ys = hand.landmarks.map((l) => l.y * height);
      const pad = 12;
      const x1 = Math.max(0, Math.min(...xs) - pad);
      const y1 = Math.max(0, Math.min(...ys) - pad);
      const x2 = Math.min(width, Math.max(...xs) + pad);
      const y2 = Math.min(height, Math.max(...ys) + pad);
      ctx.strokeStyle = color.box;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }, [hands, width, height, videoRef, shape, face]);

  return <canvas ref={canvasRef} className="task-canvas" />;
}
