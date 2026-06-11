import type { Task } from "../tasks";

const PLATFORM_ICONS: Record<string, string> = {
  Web: "💻",
  Android: "📱",
  iOS: "📱",
  Desktop: "🖥",
};

interface Props {
  task: Task;
  onStart: (task: Task) => void;
}

export function TaskCard({ task, onStart }: Props) {
  const diffClass =
    task.difficulty === "Başlangıç"
      ? "badge badge-beginner"
      : task.difficulty === "Orta"
      ? "badge badge-medium"
      : "badge badge-advanced";

  return (
    <div className="task-card">
      <div className="task-head">
        <div className="task-icon">{task.icon}</div>
        <div className="task-titles">
          <h3>{task.title}</h3>
          <div className="task-badges">
            <span className="badge badge-category">{task.category}</span>
            <span className={diffClass}>{task.difficulty}</span>
            {task.realDetector && (
              <span className="badge badge-real">✓ Gerçek dedektör</span>
            )}
            {task.experimental && (
              <span className="badge badge-exp">⚠ Deneysel</span>
            )}
          </div>
        </div>
      </div>

      <p className="task-desc">{task.description}</p>

      <div className="task-platforms">
        {task.platforms.map((p) => (
          <span key={p} className="platform">
            {PLATFORM_ICONS[p] ?? "•"} {p}
          </span>
        ))}
      </div>

      <button className="task-cta" onClick={() => onStart(task)}>
        ▶ Bu bulmacayı dene
      </button>
    </div>
  );
}
