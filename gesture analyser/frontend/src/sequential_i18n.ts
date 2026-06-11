// Sıralı Görev (mod 3) komut metinlerinin Türkçeleştirilmesi.
// PythonProject1/sequential_session.py içindeki _STEPS listesindeki SeqStep.name
// alanına göre.

export const SEQ_TR: Record<string, string> = {
  // Tek el parmak sayma
  // NOT: PythonProject1 bu adımlar için count_fingers_total (iki elin toplamı)
  // kullanıyor — yani "sol 2 + sağ 3 = 5" de kabul ediliyor. Metni gerçek
  // doğrulamayla uyumlu yapıyoruz (PythonProject1'e dokunmuyoruz).
  "Show FIST (0 fingers)":          "Yumruk yap (0 parmak)",
  "Show 1 FINGER (point)":          "Toplam 1 parmak göster",
  "Show 2 FINGERS (peace)":         "Toplam 2 parmak göster",
  "Show 3 FINGERS":                 "Toplam 3 parmak göster",
  "Show 4 FINGERS":                 "Toplam 4 parmak göster",
  "Show 5 FINGERS (open hand)":     "Toplam 5 parmak göster",
  // İki el jestleri
  "OPEN BOTH HANDS (10 fingers)":   "İki eli aç (10 parmak)",
  "BOTH FISTS (0+0)":               "İki yumruk yap (0+0)",
  "LEFT THUMB UP (1 finger left)":  "Sol başparmak yukarı",
  "LEFT 3 + RIGHT 2":               "Sol 3 + Sağ 2 parmak",
  // Hareket
  "WAVE Your Hand":                 "Elini salla",
  // Mekânsal
  "Move Hand CLOSER":               "Elini yaklaştır",
  "Move Hand AWAY":                 "Elini uzaklaştır",
  // Çizim
  "DRAW A CIRCLE":                  "Daire çiz",
  "DRAW A SQUARE":                  "Kare çiz",
  // Parmak değdirme
  "TOUCH Thumb to Index":           "Başparmağı işaret parmağına değdir",
  "TOUCH Thumb to Pinky":           "Başparmağı serçe parmağına değdir",
  // İleri
  "FLIP Your Hand (palm then back)": "Avucunu çevir (avuç → arka)",
  "PEEK-A-BOO (hide then show)":    "Ce-e (gizle, sonra göster)",
};

export function seqStepName(name: string | undefined): string {
  if (!name) return "";
  return SEQ_TR[name] ?? name;
}

export const SEQ_STATE_TR: Record<string, string> = {
  ACTIVE:       "Aktif",
  HOLDING:      "Sabit tut…",
  STEP_DONE:    "Geçti!",
  STEP_TIMEOUT: "Süre doldu",
  COMPLETE:     "Tamamlandı",
};

export const SEQ_RESULT_TR: Record<string, string> = {
  PENDING:   "•",
  PASSED:    "✓",
  TIMED_OUT: "✗",
};
