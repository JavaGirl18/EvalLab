from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class StagedExperimentHandler(ABC):
    """
    Base class for staged (multi-round) experiments.

    Subclasses implement on_stage_complete() to inspect the results of a
    finished stage and return new run-item dicts for the next stage. The
    batch executor calls this between stages and inserts the returned items
    before proceeding.

    For flat (single-stage) experiments, this class is not used — the
    existing batch_runner path applies unchanged.
    """

    @abstractmethod
    def on_stage_complete(
        self,
        stage: int,
        batch_id: str,
        experiment_id: str,
        completed_items: list[dict],
    ) -> list[dict]:
        """
        Called after all items in `stage` finish (completed or skipped).

        Args:
            stage: The stage that just finished (0-based).
            batch_id: The batch being executed.
            experiment_id: The experiment this batch belongs to.
            completed_items: All batch_run_items rows for this stage,
                             with result_json parsed into 'result' key
                             where available.

        Returns:
            List of run-item dicts for the next stage. Each dict must have:
              - item_id (str)
              - prompt_variant (str)
              - model (str)
              - stage (int) — must equal `stage + 1`
              - run_type (str)
              - depends_on (list[str] | None) — run_keys this item depends on

            Return an empty list if there are no more stages.
        """
        ...

    @abstractmethod
    def dispatch(
        self,
        run_type: str,
        item_row: dict,
        config: dict,
    ):
        """
        Called by the executor instead of the default _run_unit logic when
        run_type != 'generation'. Must return a JSON-serializable result dict.

        Args:
            run_type: The run_type value from the item row.
            item_row: The full batch_run_items row as a dict.
            config: The batch config dict.

        Returns:
            A JSON-serializable dict that will be stored in result_json.
        """
        ...
