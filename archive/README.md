# Archived experiment source

These files are verbatim copies of selected original JEPA4M experiment scripts.
Their SHA256 digests and source-relative paths appear in `../PROVENANCE.json`.

They are reviewable reference source, not a portable experiment distribution.
Imports, hashes, source-bound native classes, simulator versions, checkpoint and
cache paths, past resource receipts and experiment-tree dependencies refer to
the original frozen local study. Most of that tree is intentionally omitted.
Do not infer that running these files from this clone reproduces the original
qualification, neural predictions, native control panel, or replay results.

The archived command entrypoints are not executed by the quickstart or tests.
The portable package copies only the numerical helpers and CEM checking block
documented in the provenance manifest; it does not call the original admission
or serving paths.

Folders retain their original date-stamped names to preserve correspondence
with the evidence reports. `world_model_visual_transfer_2026_10_19` includes only
the original data adapter and feature-cache/checkpoint-contract source, not the
external model constructor or model weights.

