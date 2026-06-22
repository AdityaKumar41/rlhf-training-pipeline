"use client";

import { Check, Clock, RefreshCw, SkipForward } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

type Task = {
  task_id: string;
  prompt: string;
  response_a: string;
  response_b: string;
  metadata: Record<string, unknown>;
};

type Stats = {
  tasks: number;
  annotations: number;
  annotators: number;
  choices: Record<string, number>;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function AnnotationPage() {
  const [annotatorId, setAnnotatorId] = useState("annotator-local");
  const [task, setTask] = useState<Task | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loadedAt, setLoadedAt] = useState<number>(Date.now());
  const [status, setStatus] = useState("Ready");

  const elapsed = useMemo(() => Math.max(0, Date.now() - loadedAt), [loadedAt]);

  const loadStats = useCallback(async () => {
    const response = await fetch(`${apiBase}/api/stats`);
    if (response.ok) {
      setStats(await response.json());
    }
  }, []);

  const loadTask = useCallback(async () => {
    setStatus("Loading");
    const response = await fetch(`${apiBase}/api/tasks/next/${encodeURIComponent(annotatorId)}`);
    if (response.status === 404) {
      setTask(null);
      setStatus("Queue empty");
      await loadStats();
      return;
    }
    if (!response.ok) {
      setStatus("API unavailable");
      return;
    }
    setTask(await response.json());
    setLoadedAt(Date.now());
    setStatus("Annotating");
    await loadStats();
  }, [annotatorId, loadStats]);

  const choose = useCallback(
    async (chosen: "a" | "b" | "tie" | "skip") => {
      if (!task) return;
      setStatus("Saving");
      const response = await fetch(`${apiBase}/api/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_id: task.task_id,
          annotator_id: annotatorId,
          chosen,
          annotation_time_ms: Date.now() - loadedAt
        })
      });
      if (!response.ok) {
        setStatus("Save failed");
        return;
      }
      await loadTask();
    },
    [annotatorId, loadedAt, loadTask, task]
  );

  useEffect(() => {
    loadTask();
  }, [loadTask]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "a") choose("a");
      if (event.key.toLowerCase() === "b") choose("b");
      if (event.key.toLowerCase() === "t") choose("tie");
      if (event.key.toLowerCase() === "s") choose("skip");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [choose]);

  return (
    <main className="workspace">
      <header className="topbar">
        <div>
          <h1>Preference Annotation</h1>
          <p>{status}</p>
        </div>
        <div className="toolbar">
          <label>
            Annotator
            <input value={annotatorId} onChange={(event) => setAnnotatorId(event.target.value)} />
          </label>
          <button onClick={loadTask} aria-label="Refresh task">
            <RefreshCw size={18} />
          </button>
        </div>
      </header>

      <section className="stats">
        <span>Tasks {stats?.tasks ?? 0}</span>
        <span>Annotations {stats?.annotations ?? 0}</span>
        <span>Annotators {stats?.annotators ?? 0}</span>
        <span className="clock"><Clock size={16} /> {Math.round(elapsed / 1000)}s</span>
      </section>

      {task ? (
        <>
          <section className="prompt">
            <span>{task.task_id}</span>
            <p>{task.prompt}</p>
          </section>

          <section className="responses">
            <article>
              <div className="responseHeader">
                <strong>Response A</strong>
                <button onClick={() => choose("a")}>
                  <Check size={18} /> Choose A
                </button>
              </div>
              <p>{task.response_a}</p>
            </article>
            <article>
              <div className="responseHeader">
                <strong>Response B</strong>
                <button onClick={() => choose("b")}>
                  <Check size={18} /> Choose B
                </button>
              </div>
              <p>{task.response_b}</p>
            </article>
          </section>

          <section className="secondaryActions">
            <button onClick={() => choose("tie")}>Both good / tie</button>
            <button onClick={() => choose("skip")}>
              <SkipForward size={18} /> Skip
            </button>
          </section>
        </>
      ) : (
        <section className="empty">No annotation task is currently available.</section>
      )}
    </main>
  );
}

