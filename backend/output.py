import os
import pandas as pd
from datetime import datetime
from judge import ScoredResponse

# The folder where all CSV files will be saved.
# os.path.dirname(__file__) gets the directory of the current file (backend/).
# os.path.join() safely builds a file path that works on Mac, Windows, and Linux.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_to_csv(scored_responses: list[ScoredResponse]) -> str:
    """
    Takes a list of ScoredResponse objects and saves them as a CSV file.
    Returns the file path so the API can tell the frontend where to find it.
    """

    # Convert each ScoredResponse dataclass into a plain dict.
    # vars() is a built-in Python function that takes an object and returns
    # its fields as a dictionary — e.g. {"model": "gpt-4o", "task": "resume", ...}
    rows = [vars(r) for r in scored_responses]

    # pd.DataFrame() takes a list of dicts and turns it into a table.
    # Each dict becomes a row; the keys become column headers.
    df = pd.DataFrame(rows)

    # Build a timestamped filename so results never overwrite each other.
    # datetime.now() gets the current date and time.
    # .strftime() formats it as a string — "%Y%m%d_%H%M%S" produces e.g. "20260420_143022"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results_{timestamp}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Make sure the results directory exists before trying to write to it.
    # exist_ok=True means "don't raise an error if it already exists."
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save the DataFrame to CSV. index=False means don't write row numbers
    # as an extra column — we don't need them.
    df.to_csv(filepath, index=False)

    return filepath
