# LeWM / PushT Evaluation Harness

**DINO-WM-derived action normalization; externally pretrained LeWM tiny-ViT model.**

This research engineering portfolio studies the gap between a world model's
prediction accuracy and its closed-loop control performance. The original local
work implements action/data interfaces, frozen-checkpoint qualification, temporal
packing, CEM bookkeeping, native relative-action execution, saved-array audits,
and trajectory replay. The model itself is external LeWM, not an original DINO
or DINOv2 implementation.

The portable part of this repository runs a small offline arithmetic demo.
Archived experiment source and selected historical reports document the larger
study. Cloning this export does not reproduce the original model inference or
control experiment: weights, public dataset archives, native simulator/runtime,
feature caches, trajectories and the full hash-bound experiment tree are absent.

## Try the offline audit

Python 3.10+ is required. Only NumPy and pytest are needed; the demo never loads
Torch, downloads weights, renders images, or executes a robot environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m wm_interface_audit.demo
python -m pytest -q
```

The demo generates a declared synthetic trace with eight slots, 64 candidates
and three CEM iterations. It verifies normalization/inversion, chronological
five-command packing, RNG replay, post-clipping distributions, stable elite
selection, strict incumbent updates, and rejection of a corrupted selected ID.
The terminal embeddings come from a deterministic mock array mapping. The
1,536 checked candidate plans are a software fixture, not model performance.

`src/wm_interface_audit/extracted.py` retains the numerical CEM audit statements
and normalization helpers from the archived audit. [PROVENANCE.json](PROVENANCE.json)
records original source paths, line numbers, hashes and the limited wrapper
changes. The demo producer and tests were added for this portfolio export.
Run Python normally: the preserved auditor uses assertions and must not be run
with `python -O`.

## Historical study

The original fixed adaptation loaded an external LeWM checkpoint with
18,034,478 parameters and 303 state tensors. Its encoder is a tiny ViT with a
192-dimensional representation. DINO-WM supplies the lineage and literal action
normalization constants, not this checkpoint's encoder.

| Saved experiment | Recorded result | Meaning |
| --- | --- | --- |
| Complete control panel | 0/8 successes; 2,400 raw controls | The fixed adaptation did not solve the eight reused public validation starts. |
| Independent native replay | 2,408 checked states; maximum state error 0 | Replay consistency on the same simulator backend, not independent physics validation. |
| Observed validation transitions | Terminal latent MSE 0.0320928 versus persistence 0.9910075 | Action-conditioned prediction on 60 observed windows from 20 episodes, not task success. |

The result is a useful negative finding: lower observed latent prediction error
did not establish a control benefit. The study does not identify a single cause
of failure. The public validation starts were reused, external pretraining/data
overlap is unknown, and the action scaler is a declared DINO-statistics
adaptation rather than a recovered historical LeWM scaler. No claim of an
original-paper reproduction or a new blind benchmark is made.

The reports under [evidence/](evidence/README.md) were copied from the existing
local study. The portfolio preparation reruns only the offline synthetic audit
and its tests; it does not rerun the historical neural or control experiments.

## Repository contents

- `src/wm_interface_audit/`: portable extracted arithmetic and explicit synthetic fixture.
- `tests/`: numerical contracts, chronological packing, warm starts, tie behavior and tamper rejection.
- `archive/scripts/`: verbatim qualification, control, relative executor, data-interface and replay source for review. These depend on the omitted frozen experiment tree and are not standalone commands.
- `evidence/`: historical scalar summaries and reports, with result definitions and limitations.
- `PROVENANCE.json`: hashes of copied files and exact extracted numerical blocks.

## Model and source attribution

- **LeWM model and implementation:** [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), source revision `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; external checkpoint [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht), revision `22b330c28c27ead4bfd1888615af1340e3fe9052`.
- **DINO-WM action semantics/statistics:** [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm), pinned loader revision `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; the selected recipe uses `mu=(-0.0087, 0.0068)` and `sigma=(0.2019, 0.2002)`.
- **Local contribution:** the JEPA4M interface, qualification, execution and independent audit scripts exported here, plus the new lightweight packaging/demo/tests.

No upstream model implementation, weights, or dataset is vendored. Upstream
projects retain their own licenses; this export does not grant rights to their
models or assets. The local original tree had no project license file, so no
new open-source license has been assigned by this export.

