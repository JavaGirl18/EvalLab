# EvalLabs External Package Exchange 001

## Object

One 88.26-second spoken-audio fragment from LibriVox's *The Tao Teh King*, Chapters 8–9, together with one exact PocketSphinx ASR run, visible engine uncertainty, a source-reference transcript and an explicit difference record.

## What is preserved separately

```text
parent MP3
selected WAV evaluation source
source-reference text
ASR input derivative
raw ASR output
uncertainty record
difference record
human-review status
```

Nothing overwrites another layer.

## Important deviations and limits

1. The selected WAV is a FL Studio export derived from the LibriVox parent MP3. It is the normative evaluation source, but the export is not byte-reproducible from the recorded metadata alone.
2. LibriVox's public-domain statement is US-specific and carries a non-US caveat. This package is for the agreed private comparison and does not assert universal public redistribution rights.
3. A blind separately prepared human review has not been performed. It remains explicitly `NOT_RUN`; the published source text is not falsely relabelled as human listening evidence.
4. Loek's independent interpretation is deliberately absent so EvalLabs can write its first observations without being pulled toward ours.

## Exact run result

```text
ASR engine: PocketSphinx 5.1.1
reference words: 182
hypothesis words: 185
WER against source reference: 20.3%
low-probability engine segments: 21/185
```

## Status

```text
SOURCE AND HASHES                 PASS
EXACT LOCAL ASR RUN               PASS
RAW OUTPUT PRESERVED              PASS
VISIBLE UNCERTAINTY               PASS
SOURCE-REFERENCE DIFFERENCE       PASS
BLIND HUMAN REVIEW                NOT_RUN
PUBLIC REDISTRIBUTION AUTHORITY   NOT_CLAIMED
PRIVATE EVAL-LABS UPLOAD          READY AFTER LOEK RATIFICATION
```
