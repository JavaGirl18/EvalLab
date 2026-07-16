# Visible uncertainty

PocketSphinx supplied engine probabilities per decoded segment. This package marks 21 of 185 spoken segments below the declared threshold of 0.25.

These values are not calibrated confidence and do not prove that high-probability words are correct. The exact raw values remain in `engine_uncertainty.json` and the full segment stream remains in the ASR raw output.

Uncertainty is not silently replaced with the source text.
