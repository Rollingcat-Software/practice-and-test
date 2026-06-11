// Yüz görevleri için durum talimatları ve canlı metrik etiketleri.
// Backend task_id'ye göre kullanıcıya gösterilecek aktif talimat.

export const FACE_INSTRUCTION: Record<string, string> = {
  blink: "İki gözünü kırp",
  wink_left: "Sol gözünü kapat (sağ açık kalsın)",
  wink_right: "Sağ gözünü kapat (sol açık kalsın)",
  smile: "Geniş gülümse 😄",
  mouth_open: "Ağzını aç 😮",
  brows_up: "İki kaşını birden kaldır",
  brow_left: "Sadece sol kaşını kaldır",
  brow_right: "Sadece sağ kaşını kaldır",
  head_left: "Başını sola çevir ↪️",
  head_right: "Başını sağa çevir ↩️",
  look_up: "Yukarı bak ⬆️",
  look_down: "Aşağı bak ⬇️",
  nod_yes: "Başını salla (Evet) — yukarı/aşağı",
  shake_no: "Başını salla (Hayır) — sağa/sola",
};

export function faceInstruction(taskId: string | undefined): string {
  if (!taskId) return "";
  return FACE_INSTRUCTION[taskId] ?? taskId;
}

// Bazı görevler için canlı metrik etiketi (HUD)
export const FACE_METRIC_LABEL: Record<string, string> = {
  blink: "Göz kapanma",
  wink_left: "Sol göz",
  wink_right: "Sağ göz",
  smile: "Gülümseme",
  mouth_open: "Ağız açıklığı",
  brows_up: "Kaş kaldırma",
  brow_left: "Sol kaş",
  brow_right: "Sağ kaş",
  head_left: "Sola dönüş",
  head_right: "Sağa dönüş",
  look_up: "Yukarı eğim",
  look_down: "Aşağı eğim",
  nod_yes: "Dönüş sayısı",
  shake_no: "Dönüş sayısı",
};

export function faceMetricLabel(taskId: string | undefined): string {
  if (!taskId) return "Sinyal";
  return FACE_METRIC_LABEL[taskId] ?? "Sinyal";
}

export const FACE_STATE_TR: Record<string, string> = {
  NO_FACE: "Yüz bekleniyor…",
  ARMING: "Hazırlanıyor — yüzünü düz tut",
  ACTIVE: "Hareketi yap",
  SUCCESS: "Doğrulandı!",
};
