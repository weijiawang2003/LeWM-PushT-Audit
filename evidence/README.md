# Historical evidence and result definitions

These reports and scalar summaries were copied from the pre-existing local
JEPA4M study. They are historical evidence, not outputs from the portable demo.
Copied file hashes are recorded in `../PROVENANCE.json`. Full datasets, images,
feature caches, predictions, raw trajectories, weights and runtime receipts are
not distributed, so this export does not independently reproduce their values.

## What was measured

**Control:** `control/SUMMARY.json` reports a completed fixed panel of eight
public, previously used validation seeds (`88000` through `88007`) for one T
goal. Each episode allows 300 raw controls, replanning every five controls.
Success requires native and separately calculated polygon coverage both
strictly above 0.95. All eight episodes ran 300 controls without reaching the
threshold, hence 0/8 and 2,400 controls. The original driver distinguishes
incomplete/aborted panels from completed failures.

The saved independent audit replayed those controls through the same native
simulator backend: 2,408 states and 480 decision RGB hashes were checked, with
maximum state error 0. This is consistency evidence rather than a second physics
implementation. The complete original control driver, relative executor and
saved audit source are included under `../archive/scripts/` for review.

**Observed prediction:** `transition/SUMMARY.json` defines the primary case as
H1/F3 terminal latent MSE at raw `t+15`. Each block contains five chronological
2D commands, the predicted state is a 192-dimensional E embedding, and the
comparison averages terminal squared error over 60 windows from 20 validation
episodes. True-action MSE is `0.032092783600091934`; persistence MSE is
`0.9910075068473816`. Wrong-action and zero-action predictions are scored
against the same observed future, not measured counterfactual futures.

The reported 96.76% latent MSE reduction is not geometric-state accuracy,
control success, a JEPA mechanism proof, or a generalization result with known
pretraining exclusion. It is compatible with the separate 0/8 control result.

**Qualification:** `qualification/SUMMARY.json` records full strict checkpoint
loading, unchanged frozen state/BatchNorm hashes, cached/original rollout
agreement and interface/cost checks. That stage has no task-performance score.
Its LeWM model uses a tiny ViT encoder, not DINOv2. DINO-WM contributes the
chosen normalization constants and action-semantics source lineage.

`DINO_SOURCE_STATS_ADAPTATION_ADDENDUM.md` explains the declared adaptation and
why it is not a reconstruction of the historical LeWM scaler. Original reports
retain detailed resource accounting and limits; quoted CPU timings apply to
their stated scopes and are not a complete project/pretraining cost certificate.
