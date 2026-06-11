// Yüz Görevleri — 14 baş/yüz hareketi kartı.
// Backend'de PythonProject1'e dokunmadan eklenen FaceTracker + FaceTaskSession
// tarafından işlenir (MediaPipe FaceLandmarker + blendshape'ler).

import type { Task } from "./tasks";

const PLATFORMS_WEBONLY: Task["platforms"] = ["Web", "Android", "iOS", "Desktop"];

export const FACE_TASKS: Task[] = [
  {
    id: "face-blink",
    faceTaskId: "blink",
    family: "face",
    title: "Göz Kırp",
    description:
      "Bir kez hızlı ve doğal şekilde göz kırpın — iki gözünüzü kısaca kapatıp tekrar açın. Gözleriniz yeniden açıldığı an tamamlanır; kapalı tutmanıza gerek yok.",
    difficulty: "Başlangıç",
    category: "Yüz",
    icon: "👁",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-wink-left",
    faceTaskId: "wink_left",
    family: "face",
    title: "Sol Gözü Kapat",
    description:
      "Sol gözünüzü bir kez kırpın — sağ göz açık kalırken solu kısaca kapatıp tekrar açın. Sol göz açık kalırken solu kapatın.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "😉",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-wink-right",
    faceTaskId: "wink_right",
    family: "face",
    title: "Sağ Gözü Kapat",
    description:
      "Sağ gözünüzü bir kez kırpın — sol göz açık kalırken sağı kısaca kapatıp tekrar açın. Göz yeniden açıldığı an tamamlanır.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "😉",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-smile",
    faceTaskId: "smile",
    family: "face",
    title: "Geniş Gülümse",
    description:
      "Geniş bir şekilde gülümseyin (dişler görünüyorsa daha iyi). Dudak köşesi açıklığını ölçer.",
    difficulty: "Başlangıç",
    category: "Yüz",
    icon: "😄",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-mouth-open",
    faceTaskId: "mouth_open",
    family: "face",
    title: "Ağzı Aç",
    description:
      "Ağzınızı geniş bir şekilde açın. Yüz yüksekliğine göre dudak arası mesafeyi ölçer.",
    difficulty: "Başlangıç",
    category: "Yüz",
    icon: "😮",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-head-left",
    faceTaskId: "head_left",
    family: "face",
    title: "Başı Sola Çevir",
    description:
      "Başınızı solunuza çevirin. İşaret noktalarından baş-poz yaw açısını takip eder.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "↪️",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-head-right",
    faceTaskId: "head_right",
    family: "face",
    title: "Başı Sağa Çevir",
    description:
      "Başınızı sağınıza çevirin. İşaret noktalarından baş-poz yaw açısını takip eder.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "↩️",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-look-up",
    faceTaskId: "look_up",
    family: "face",
    title: "Yukarı Bak",
    description:
      "Çenenizi yukarı kaldırın. İşaret noktalarından baş-poz pitch açısını takip eder.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "⬆️",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-look-down",
    faceTaskId: "look_down",
    family: "face",
    title: "Aşağı Bak",
    description:
      "Çenenizi aşağı indirin. İşaret noktalarından baş-poz pitch açısını takip eder.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "⬇️",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-brows-up",
    faceTaskId: "brows_up",
    family: "face",
    title: "Her İki Kaşı Kaldır",
    description:
      "Her iki kaşınızı birlikte kaldırın. Kaş-göz mesafesi değişimini ölçer.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "😯",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-brow-left",
    faceTaskId: "brow_left",
    family: "face",
    title: "Sol Kaşı Kaldır",
    description:
      "Sadece sol kaşınızı kaldırın. Asimetrik kaş kontrolünü test eder.",
    difficulty: "İleri",
    category: "Yüz",
    icon: "🤨",
    realDetector: true,
    experimental: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-brow-right",
    faceTaskId: "brow_right",
    family: "face",
    title: "Sağ Kaşı Kaldır",
    description:
      "Sadece sağ kaşınızı kaldırın. Asimetrik kaş kontrolünü test eder.",
    difficulty: "İleri",
    category: "Yüz",
    icon: "🤨",
    realDetector: true,
    experimental: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-nod-yes",
    faceTaskId: "nod_yes",
    family: "face",
    title: "Başı Salla (Evet)",
    description:
      "Başınızı yukarı-aşağı sallayın. Kısa bir pencerede pitch hareketini izler.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "🙆",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
  {
    id: "face-shake-no",
    faceTaskId: "shake_no",
    family: "face",
    title: "Başı Salla (Hayır)",
    description:
      "Başınızı sağa-sola sallayın. Kısa bir pencerede yaw hareketini izler.",
    difficulty: "Orta",
    category: "Yüz",
    icon: "🙅",
    realDetector: true,
    platforms: PLATFORMS_WEBONLY,
  },
];
