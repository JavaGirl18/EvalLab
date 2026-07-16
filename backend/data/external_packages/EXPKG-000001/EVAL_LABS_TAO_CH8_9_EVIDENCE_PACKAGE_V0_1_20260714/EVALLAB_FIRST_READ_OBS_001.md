# EvalLab Independent First-Read Observation

**Package:** EXPKG-000001  
**Evaluation Record:** EV-2026-000003  
**Observer:** Valencia Cooper, EvalLab  
**Date:** 2026-07-15  
**Status:** FROZEN — recorded before receiving Loek's interpretation

---

## Comparison question addressed

*How does each artifact structure represent the same AI-assisted transcription transformation, and what does each preserve, separate, collapse or leave unresolved?*

---

## Observation 01 — Raw ASR Output

*Artifact: 04_ASR_RUN*

The ASR output is preserved exactly as produced by the recognition engine and is explicitly identified as a hypothesis rather than a definitive transcript. This distinction is important: it frames the output as the model's best interpretation of the audio, not as an authoritative representation of what was spoken.

Inspection shows the hypothesis contains numerous lexical substitutions and phrases that are semantically incoherent, indicating substantial information distortion. Rather than correcting these errors, the package preserves them as primary evidence. This allows subsequent review, comparison, and analysis to be performed against the original machine output without losing the provenance of the transformation.

**Takeaway:** The methodology prioritizes preservation of the machine's original interpretation over producing an immediately readable transcript. The raw model output is treated as evidence rather than corrected or normalized before review.

---

## Observation 02 — Semantic Severity Is Invisible to WER

*Artifacts: 07_DIFFERENCE, 03_REFERENCE_TEXT, 04_ASR_RUN*

The 20.3% WER figure treats all word-level errors as equivalent. EvalLab's taxonomy does not.

**From the files:**

| Reference | ASR Hypothesis | WER count | Notes |
|---|---|---|---|
| "gold and jade fill the hall" | "colin jade fill the hole" | 2 substitutions | Incoherent text that has no relevance to what the speaker said |
| "does not wrangle about his low position" | "does not bring goal but his low position" | 2 substitutions + 1 insertion | Fabrication — completely changes the context of the sentence |
| "leave a vessel unfilled" | "leave a vessel on the field" | 2 substitutions | Fabrication — completely changes the context of the sentence |

**Takeaway:** WER is a measurement of phonetic fidelity. It does not distinguish between errors that are cosmetic and errors that are philosophically significant. EvalLab's framework separates these by dimension and severity. That is a structural difference between the two approaches.

---

## Observation 03 — Engine Uncertainty as a First-Class Signal

*Artifact: 05_UNCERTAINTY*

The 21 low-probability segments are preserved as a named layer, not discarded or averaged into the WER score. The engine's own doubt about its output is treated as evidence alongside the output itself.

EvalLab currently carries judge confidence as a signal — but that is the judge's confidence in its *assessment*, not the subject model's confidence in its *output*. This package surfaces something EvalLab does not yet capture: the subject model flagging its own uncertainty at the moment of production.

**Takeaway:** Subject model output confidence and judge assessment confidence are different signals. The package treats them as such. This is a methodological gap worth noting for EvalLab's own pipeline.

---

## Observation 04 — Structural Principle

The package intentionally separates measurement from interpretation. The WER figure, the uncertainty record, and the difference alignment are all measurement artifacts. No interpretation is offered alongside them.

This mirrors EvalLab's own design principle of separating the evaluation record from the analyst's conclusions — but applies it upstream, at the point of artifact creation rather than at the point of evaluation.

**Takeaway:** This is a shared structural commitment, arrived at independently.

---

## What remains unresolved

- Whether individual errors are acoustic failures (recording quality, room conditions) or vocabulary failures (model not trained on archaic Taoist vocabulary). The evidence cannot distinguish these without additional analysis.
- Whether the FL Studio WAV derivation introduces any acoustic artifact that affected the ASR run. The TRANSFORMATION_RECORD is preserved but the derivation is not byte-reproducible.
- Human listening review is explicitly NOT_RUN. The 20.3% WER is scored against the Legge source text, not against what the reader actually said. Any reading slips remain an open layer.
- No metadata about the LibriVox reader or recording conditions. This limits interpretation of the absolute WER figure — the difficulty of the audio is unknown.

---

*This document represents EvalLab's first observations, frozen before exchange with Loek's interpretation. Differences between the two readings are expected and intentional per the independent first-read protocol.*
