import { useState } from "react";
import { HAND_TASKS, type Task } from "./tasks";
import { FACE_TASKS } from "./face_tasks";
import { TaskCard } from "./components/TaskCard";
import { TaskRunner } from "./components/TaskRunner";

export default function App() {
  const [active, setActive] = useState<Task | null>(null);

  if (active) {
    return <TaskRunner task={active} onClose={() => setActive(null)} />;
  }

  return (
    <div className="app">
      {/* El Hareketi Görevleri */}
      <section className="section-card">
        <header className="section-head">
          <div className="section-icon">✋</div>
          <div className="section-meta">
            <div className="section-title-row">
              <h2>El Hareketi Görevleri</h2>
              <span className="count-pill">{HAND_TASKS.length}</span>
            </div>
            <p>
              MediaPipe HandLandmarker ile takip edilir — parmak sayısı,
              sıkıştırma, şekil çizimi ve fazlası.
            </p>
          </div>
        </header>

        <div className="task-grid">
          {HAND_TASKS.map((t) => (
            <TaskCard key={t.id} task={t} onStart={setActive} />
          ))}
        </div>
      </section>

      {/* Yüz Görevleri */}
      <section className="section-card section-face">
        <header className="section-head">
          <div className="section-icon section-icon-face">😊</div>
          <div className="section-meta">
            <div className="section-title-row">
              <h2>Yüz Görevleri</h2>
              <span className="count-pill count-pill-face">
                {FACE_TASKS.length}
              </span>
            </div>
            <p>
              Her görev için ayrı ayarlı — göz kırpma, gülümseme, başını çevirme.
              Her kart gerçek BlazeFace + MediaPipe FaceLandmarker akışını çalıştırır.
            </p>
          </div>
        </header>

        <div className="task-grid">
          {FACE_TASKS.map((t) => (
            <TaskCard key={t.id} task={t} onStart={setActive} />
          ))}
        </div>
      </section>

      <button className="fab" title="Sıralama" aria-label="Sıralama">
        ▦
      </button>
    </div>
  );
}
