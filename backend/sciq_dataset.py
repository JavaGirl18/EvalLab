from __future__ import annotations

_cache: dict[str, dict] | None = None


def _load_dataset() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache

    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test")
    _cache = {}
    for i, row in enumerate(ds):
        item_id = f"sciq_{i:04d}"
        _cache[item_id] = {
            "item_id":       item_id,
            "question":      row["question"],
            "support":       row["support"],
            "correct_answer": row["correct_answer"],
            "distractor1":   row["distractor1"],
            "distractor2":   row["distractor2"],
            "distractor3":   row["distractor3"],
        }
    return _cache


def load_sciq_item(item_id: str) -> dict:
    ds = _load_dataset()
    if item_id not in ds:
        raise ValueError(f"SciQ item not found: {item_id}")
    return ds[item_id]


def list_sciq_item_ids(seed: int = 42, limit: int = 100) -> list[str]:
    """Return `limit` item IDs in a fixed shuffled order for reproducible sampling."""
    import random
    ids = list(_load_dataset().keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:limit]
