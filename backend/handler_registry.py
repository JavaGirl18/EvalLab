from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment_handler import StagedExperimentHandler

_registry: dict[str, "StagedExperimentHandler"] = {}


def register_handler(experiment_type: str, handler: "StagedExperimentHandler") -> None:
    _registry[experiment_type] = handler


def get_handler(experiment_type: str | None) -> "StagedExperimentHandler | None":
    if not experiment_type:
        return None
    return _registry.get(experiment_type)
