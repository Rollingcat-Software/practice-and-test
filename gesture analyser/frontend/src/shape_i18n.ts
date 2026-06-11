// Shape Eval (Şekil Çiz) modu Türkçeleştirme.
// PythonProject1/shape_tracer.py TracerState + ShapeType.label

export const SHAPE_TR: Record<string, string> = {
  CIRCLE: "Daire",
  SQUARE: "Kare",
  TRIANGLE: "Üçgen",
  "S-CURVE": "S-Eğrisi",
};

export function shapeName(label: string | undefined): string {
  if (!label) return "şekil";
  return SHAPE_TR[label] ?? label;
}

// Tracer durumuna göre kullanıcıya gösterilecek talimat.
export function tracerInstruction(
  tracerState: string | null | undefined,
  shapeLabel: string | undefined,
): { text: string; tone: "info" | "ok" | "fail" | "go" } {
  const shape = shapeName(shapeLabel);
  switch (tracerState) {
    case "INSTRUCTING":
      return { text: `Hazırlan: ${shape} çizeceksin`, tone: "info" };
    case "IDLE":
      return {
        text: "İşaret parmağını mavi BAŞLA halkasına götür",
        tone: "info",
      };
    case "POSITIONING":
      return { text: "Sabit tut… (0.5 sn)", tone: "info" };
    case "TRACING":
      return {
        text: `${shape} şeklini çiz — bitince ✊ yumruk yap veya süreyi bekle`,
        tone: "go",
      };
    case "COMPLETED":
      return { text: "Hesaplanıyor…", tone: "info" };
    case "VERIFIED":
      return { text: "✓ DOĞRULANDI", tone: "ok" };
    case "FAILED":
      return { text: "✗ BAŞARISIZ — tekrar dene", tone: "fail" };
    default:
      return { text: shape, tone: "info" };
  }
}
