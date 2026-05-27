# FIVUCSAS — Research & Experiments

R&D, prototyping, and experimentation monorepo for [FIVUCSAS](https://github.com/Rollingcat-Software/FIVUCSAS) — **research code, not production**.

This repository collects the standalone prototypes, learning projects, and
exploratory spikes that fed into the FIVUCSAS biometric identity platform:
face recognition pipelines, liveness / anti-spoofing experiments, gesture and
challenge research, and native NFC document readers. Each subproject is
self-contained with its own dependencies and documentation. Code that graduated
to production has since been extracted into dedicated repositories (see the
links below); what remains here is the historical R&D record.

## Subprojects

| Directory | Purpose |
|-----------|---------|
| `biometric-demo-optimized/` | Biometric demo app built on a clean Hexagonal Architecture — MediaPipe face detection, 468-point landmarks, two-phase enrollment, and a 14-challenge biometric puzzle. |
| `DeepFacePractice1/` | Learning-oriented DeepFace project for face verification, attribute analysis, recognition search, and embeddings, organized with a clean domain/architecture layout. |
| `GestureAnalysis/` | Gesture, finger-touch, and motion experiments — active-liveness management, hand tracking, and anti-spoof challenge validation. |
| `liveness-antispoof-research/` | Archival collection of liveness / anti-spoof R&D branches and running examples. Production algorithms now live in [spoof-detector](https://github.com/Rollingcat-Software/spoof-detector). |
| `TurkishEidNfcReader/` | Android (Kotlin + Jetpack Compose) NFC reader for the Turkish National eID card over ISO 7816-4 / IsoDep — PIN auth, DG1/DG2 reading, SOD validation. |
| `UniversalNfcReader/` | Modular Android NFC reader supporting multiple card types (Turkish eID, Istanbulkart, MIFARE Classic/DESFire/Ultralight, ISO 15693, NDEF) via a factory pattern. |

Dated planning, status, and compliance reports are kept under
[`docs/archive/`](docs/archive/).

## ⚠️ Research / experimental — not production

The code here is **experimental and may be incomplete**. It is intended for
research, learning, and reference only. Do **not** deploy any of it as-is, and
do not treat it as a source of production-grade or security-reviewed code.

## Related repositories

- **Parent product:** <https://github.com/Rollingcat-Software/FIVUCSAS>
- **Production anti-spoof library:** <https://github.com/Rollingcat-Software/spoof-detector>

## License

[MIT](LICENSE) © 2026 FIVUCSAS Contributors
