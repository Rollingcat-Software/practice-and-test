// Liveness komutlarının Türkçeleştirilmesi.
// PythonProject1/liveness_session.py içindeki ALL_COMMANDS listesindeki ham
// İngilizce metinlerin (Command.name) okunabilir Türkçe karşılıkları.

export const COMMAND_TR: Record<string, string> = {
  // Gesture komutları
  // NOT: PythonProject1/liveness_session.py içindeki komut metinleri ile gerçek
  // doğrulama (gesture_reqs) bazı durumlarda uyumsuz — örn. "SHOW 2 FINGERS!"
  // arka planda yalnızca SAĞ elde 2 parmak kontrolü yapıyor. PythonProject1'e
  // dokunmadığımız için, frontend çevirisini gerçek doğrulamayla uyumlu
  // yapıyoruz ki kullanıcı doğru elini gösterip geçebilsin.
  "SHOW LEFT FIST!": "SOL YUMRUK GÖSTER!",                  // Left=0   ✓
  "SHOW RIGHT FIST!": "SAĞ YUMRUK GÖSTER!",                 // Right=0  ✓
  "SHOW 2 FINGERS!": "SAĞ ELDE 2 PARMAK GÖSTER!",           // Right=2  (el belirtildi)
  "SHOW 3 FINGERS!": "SAĞ ELDE 3 PARMAK GÖSTER!",           // Right=3  (el belirtildi)
  "SHOW 4 FINGERS!": "SOL ELDE 4 PARMAK GÖSTER!",           // Left=4   (el belirtildi)
  "OPEN BOTH HANDS!": "İKİ ELİ AÇ!",                        // L=5,R=5  ✓
  "LEFT THUMB UP!": "SOL BAŞPARMAK YUKARI!",                // Left=1   ✓
  "RIGHT OPEN HAND!": "SAĞ ELİ AÇ!",                        // Right=5  ✓
  // Spatial
  "MOVE HAND CLOSER!": "ELİ YAKLAŞTIR!",
  "MOVE HAND AWAY!": "ELİ UZAKLAŞTIR!",
  // Motion
  "WAVE YOUR HAND!": "ELİNİ SALLA!",
  // Advanced
  "TOUCH THUMB TO PINKY!": "BAŞPARMAĞI SERÇE PARMAĞA DEĞDİR!",
  "TOUCH THUMB TO INDEX!": "BAŞPARMAĞI İŞARET PARMAĞINA DEĞDİR!",
  "FLIP YOUR HAND!": "AVUCUNU ÇEVİR!",
  "PEEK-A-BOO! (HIDE THEN SHOW)": "CE-E! (GİZLE, SONRA GÖSTER)",
  // Finger touch (sıkıştırma)
  "PINCH: THUMB + INDEX": "SIKIŞTIR: BAŞPARMAK + İŞARET",
  "PINCH: THUMB + MIDDLE": "SIKIŞTIR: BAŞPARMAK + ORTA",
  "PINCH: THUMB + RING": "SIKIŞTIR: BAŞPARMAK + YÜZÜK",
  "PINCH: THUMB + PINKY": "SIKIŞTIR: BAŞPARMAK + SERÇE",
  "TOUCH BOTH THUMBS!": "İKİ BAŞPARMAĞI DEĞDİR!",
  // Shape trace
  "TRACE THE SHAPE!": "ŞEKLİ ÇİZ!",
};

export function translateCommand(name: string | undefined | null): string {
  if (!name) return "";
  return COMMAND_TR[name] ?? name;
}

export const STATE_TR: Record<string, string> = {
  ACTIVE: "Aktif",
  DEBOUNCE: "Doğrulanıyor…",
  SUCCESS: "Başarılı!",
  FAILED: "Başarısız",
  VERIFIED_100: "Doğrulama tamamlandı (%100)",
};
