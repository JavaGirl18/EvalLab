import yaml
from pathlib import Path

TAXONOMY_PATH = Path(__file__).parent / "data" / "taxonomy.yaml"


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def get_categories(taxonomy: dict) -> list[dict]:
    return taxonomy["categories"]


def get_by_meta_category(taxonomy: dict, meta_category: str) -> list[dict]:
    return [c for c in taxonomy["categories"] if c["meta_category"] == meta_category]


def compute_aggregate_scores(dimension_scores: dict, taxonomy: dict) -> tuple[float, float]:
    """
    Returns (information_integrity_score, cultural_fidelity_score), each 0–10.
    Higher = fewer / less severe failures.
    Formula: 10 - (sum_of_severities / max_possible_severity) * 10
    Category counts derive from the taxonomy, so scores stay correct as the taxonomy evolves.
    """
    def _score(cat_ids: list[str]) -> float:
        if not cat_ids:
            return 10.0
        total = sum(dimension_scores.get(cid, {}).get("severity", 0) for cid in cat_ids)
        max_possible = 3 * len(cat_ids)
        return round(10 - (total / max_possible) * 10, 1)

    ii_ids = [c["id"] for c in get_by_meta_category(taxonomy, "information_integrity")]
    cf_ids = [c["id"] for c in get_by_meta_category(taxonomy, "cultural_fidelity")]

    return _score(ii_ids), _score(cf_ids)
