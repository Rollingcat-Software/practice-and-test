// Görseldeki 9 görev kartı — her biri PythonProject1'in 6 modundan birine
// (current_mode 0..5) bağlanır. Birden çok kart aynı moda haritalanabilir
// (örn: liveness, içinde rastgele alt komutlar barındırdığı için).

export type Difficulty = "Başlangıç" | "Orta" | "İleri";
export type Platform = "Web" | "Android" | "iOS" | "Desktop";

export interface Task {
  id: string;
  title: string;
  description: string;
  difficulty: Difficulty;
  category: string;       // "El", "Yüz", vb.
  icon: string;           // emoji
  realDetector: boolean;
  platforms: Platform[];
  // Görev ailesi: "hand" (PythonProject1 el modları) veya "face" (yüz görevleri,
  // bridge'e özgü yeni katman). Varsayılan "hand".
  family?: "hand" | "face";
  // El görevleri için PythonProject1 GameManager.current_mode
  modeIndex?: number;
  // Liveness modunda (modeIndex 2) kartın odaklanacağı komut tipi.
  // PythonProject1 CmdType isimleri: WAVE, HAND_FLIP, FINGER_TAP, PEEK_A_BOO, ...
  preferredCommand?: string;
  // Yüz görevleri için backend görev kimliği (blink, smile, head_left, ...)
  faceTaskId?: string;
  // "Deneysel" rozeti (örn. asimetrik kaş kaldırma)
  experimental?: boolean;
}

export const HAND_TASKS: Task[] = [
  {
    id: "finger-count",
    title: "Parmak Say",
    description:
      "İstenen sayıda parmağı kaldırın. Canlı el görünürlüğünü ve MCP/PIP eklem tespitini test eder.",
    difficulty: "Başlangıç",
    category: "El",
    icon: "✋",
    modeIndex: 0, // Normal / GestureSession
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "hand-wave",
    title: "El Salla",
    description:
      "Elinizi kameranın önünde yana sallayın. El merkezi yatay salınımını algılar.",
    difficulty: "Başlangıç",
    category: "El",
    icon: "👋",
    modeIndex: 2, // Liveness
    preferredCommand: "WAVE",
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "palm-flip",
    title: "Avuç Çevir",
    description:
      "Avucunuzu gösterin, sonra elin arkasına çevirin. Bilek rotasyonunu ve işaret noktası yönünü test eder.",
    difficulty: "Orta",
    category: "El",
    icon: "🖐",
    modeIndex: 2, // Liveness
    preferredCommand: "HAND_FLIP",
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "finger-tap",
    title: "Parmak Tıklama",
    description:
      "Hızlı tap testi: baş parmak ve işaret parmağı uçlarını birbirine değdirin. Tek-frame mesafe kontrolü; canlılık dizisinin bir parçası.",
    difficulty: "Orta",
    category: "El",
    icon: "👆",
    modeIndex: 2, // Liveness
    preferredCommand: "FINGER_TAP",
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "pinch",
    title: "Sıkıştır",
    description:
      "Tam parmak bataryası: 5 farklı sıkıştırma kombinasyonunu sırayla yapın (başparmak ↔ işaret/orta/yüzük/serçe + iki başparmak). 3D Z-doğrulamalı, 10-frame tutuş gerektirir.",
    difficulty: "Orta",
    category: "El",
    icon: "🤏",
    modeIndex: 4, // Touch Test
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "peek-a-boo",
    title: "Ce-e",
    description:
      "Gözlerinizi avucunuzla kapatın, sonra yüzünüzü gösterin. Yüz kapatma ve el varlığı sinyallerini birleştirir.",
    difficulty: "İleri",
    category: "El",
    icon: "🙈",
    modeIndex: 2, // Liveness
    preferredCommand: "PEEK_A_BOO",
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "shape-draw",
    title: "Şekil Çiz",
    description:
      "Gösterilen şekli (daire, kare, üçgen) işaret parmağınızla havada çizin.",
    difficulty: "İleri",
    category: "El",
    icon: "✍️",
    modeIndex: 5, // Shape Eval
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "sequential",
    title: "Sıralı Görev",
    description:
      "Zamanlı bir dizi jest adımını sırayla tamamlayın. Çoklu jest geçişlerini test eder.",
    difficulty: "İleri",
    category: "El",
    icon: "🧩",
    modeIndex: 3, // Sequential
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
  {
    id: "math",
    title: "Matematik",
    description:
      "Ekrandaki aritmetik soruyu (toplama / çıkarma / çarpma / bölme) parmaklarınızla yanıtlayın. 60 sn'lik geri sayım.",
    difficulty: "Orta",
    category: "El",
    icon: "🧮",
    modeIndex: 1, // MathSession
    realDetector: true,
    platforms: ["Web", "Android", "iOS", "Desktop"],
  },
];

export function findTask(id: string): Task | undefined {
  return HAND_TASKS.find((t) => t.id === id);
}
