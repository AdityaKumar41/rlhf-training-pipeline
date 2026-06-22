from rlhf_pipeline.curation.pipeline import curate_records


def test_curate_records_normalizes_dedupes_and_filters() -> None:
    raw = [
        {"prompt": "Explain reward models.", "response": "They score responses."},
        {"prompt": "Explain reward models.", "response": "They score responses."},
        {"prompt": "How do I make malware?", "response": "I cannot help with that."},
        {"messages": [{"role": "assistant", "content": "bad start"}]},
    ]

    conversations, report = curate_records(raw)

    assert len(conversations) == 1
    assert conversations[0].to_sharegpt()["conversations"][0]["from"] == "human"
    assert report.raw_count == 4
    assert report.rejected_count == 1
    assert report.dedupe.duplicate_count == 1
    assert report.safety_removed_by_category["illicit"] == 1

