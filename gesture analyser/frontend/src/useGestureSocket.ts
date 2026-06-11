import { useEffect, useRef, useState, useCallback } from "react";

export type Landmark = { x: number; y: number; z: number };
export type Hand = { handedness: "Left" | "Right"; landmarks: Landmark[] };

export type FaceGroups = {
  oval?: number[][];
  left_eye?: number[][];
  right_eye?: number[][];
  lips?: number[][];
  left_brow?: number[][];
  right_brow?: number[][];
};

export interface StateMessage {
  type: "state";
  mode: number;
  mode_name: string;
  hands_present?: boolean;
  hands?: Hand[];
  session: Record<string, any>;
  fps: number;
  // Yüz görevleri için (mode -1):
  face_present?: boolean;
  face_groups?: FaceGroups;
}

export interface ReadyMessage {
  type: "ready";
  modes: string[];
}

export type ServerMessage =
  | StateMessage
  | ReadyMessage
  | { type: "error"; message: string }
  | { type: "mode_changed"; mode: number; mode_name: string }
  | { type: "restarted"; mode: number; mode_name: string };

type Status = "idle" | "connecting" | "open" | "closed" | "error";

export function useGestureSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [lastState, setLastState] = useState<StateMessage | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return;
    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setStatus("error");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as ServerMessage;
        if (msg.type === "state") setLastState(msg);
        else if (msg.type === "error") setLastError(msg.message);
      } catch {
        // yoksay
      }
    };
  }, [url]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setStatus("closed");
    }
  }, []);

  const send = useCallback((obj: object) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(obj));
    return true;
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { status, lastState, lastError, connect, disconnect, send };
}
