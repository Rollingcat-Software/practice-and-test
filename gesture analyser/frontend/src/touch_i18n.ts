// Touch Test (Sıkıştır) komutlarının Türkçeleştirilmesi.
// PythonProject1/finger_touch_session.py içindeki TouchCommand isimlerine göre.

export const TOUCH_TR: Record<string, { title: string; hint: string }> = {
  THUMB_TO_INDEX: {
    title: "Başparmak + İşaret",
    hint: "Baş parmak ucunu işaret parmağı ucuna değdir (aynı el)",
  },
  THUMB_TO_MIDDLE: {
    title: "Başparmak + Orta",
    hint: "Baş parmak ucunu orta parmak ucuna değdir (aynı el)",
  },
  THUMB_TO_RING: {
    title: "Başparmak + Yüzük",
    hint: "Baş parmak ucunu yüzük parmağı ucuna değdir (aynı el)",
  },
  THUMB_TO_PINKY: {
    title: "Başparmak + Serçe",
    hint: "Baş parmak ucunu serçe parmağı ucuna değdir (aynı el)",
  },
  DOUBLE_THUMB_TOUCH: {
    title: "İki Başparmak",
    hint: "İki elin baş parmak uçlarını birbirine değdir",
  },
};

export function touchTitle(name: string | undefined): string {
  if (!name) return "";
  return TOUCH_TR[name]?.title ?? name;
}

export function touchHint(name: string | undefined): string {
  if (!name) return "";
  return TOUCH_TR[name]?.hint ?? "";
}

export const TOUCH_STATE_TR: Record<string, string> = {
  ACTIVE: "Bekleniyor",
  SUCCESS: "Doğrulandı!",
  COMPLETE: "Tamamlandı",
};
