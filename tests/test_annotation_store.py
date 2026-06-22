from rlhf_pipeline.api.database import AnnotationStore
from rlhf_pipeline.schemas import AnnotationEvent, AnnotationTask


def test_annotation_store_roundtrip(tmp_path) -> None:
    store = AnnotationStore(tmp_path / "annotations.db")
    task = AnnotationTask(
        task_id="task-1",
        prompt="Pick the better response.",
        response_a="A clear answer.",
        response_b="A vague answer.",
    )
    store.upsert_task(task)

    assert store.next_task("ann-1") == task

    store.record_event(
        AnnotationEvent(
            task_id="task-1",
            annotator_id="ann-1",
            chosen="a",
            annotation_time_ms=1200,
        )
    )

    assert store.next_task("ann-1") is None
    stats = store.stats()
    assert stats["tasks"] == 1
    assert stats["annotations"] == 1
    assert stats["choices"]["a"] == 1

