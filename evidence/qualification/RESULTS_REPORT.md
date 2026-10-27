# Full pretrained LeWM: interface and complete-query cost qualification

The retained full checkpoint passed strict loading with303 state tensors and18,034,478 parameters. Encoder, projector, action encoder, predictor and prediction projector all remained present, eval/frozen, with byte-identical before/after state hashes including BatchNorm. Actual inference used MPS BF16 autocast over FP32 weights. This is the one source-fixed DINO-statistics adaptation, not reconstruction of the historical LeWM scaler or an original-paper reproduction.

Both nonconstant temporal fixtures passed: initialH1/F3/HS3 produced four frames; H3/F3/HS3 produced six with two synthetic past-action markers and three future blocks. The original rollout and cached-history helper had **zero maximum and RMSE difference**. All eight original goal embeddings reencoded with zero error; scoring still used the original cached bank. The independent saved-array audit recomputed normalization/clip/inversion, chronological packing, every saved cost, CEM ranking/update/incumbent and persistent RNG trace. It did not reload Torch/weights or reproduce a model forward.

The relative-native executor passed90 real raw controls (900 source-expected microsteps), including outside-old-box targets and independent target-equation/native-reference/relative-replay state equality. This shares the native simulator and is not independent physics. It verifies `target=current_pusher+100*u` inside the executor at every raw step, without absolute clipping or giving true positions to the model. These were fixed engineering commands on97000/97001, with no learned policy episode or performance selection.

| Complete B8 query | Process CPU seconds | Synchronized wall seconds |
|---|---:|---:|
| 0 | 0.106585 | 0.845473 |
| 1 | 0.053328 | 0.217566 |
| 2 | 0.055027 | 0.232173 |
| 3 | 0.053904 | 0.210440 |

Each query includes eight B1 image/E encodes, CEM64×3/elite8 with three relative2D knots repeated five raw commands, all512-row predictions, float32 eight-goal costs, ranking/updates and real uncompressed payload serialization/hash/memory checks. Only the final timing-sidecar write is outside its query clock; the outer guard charges it. Group cost is not B1 latency. These four engineering queries have no task score.

The predeclared forecast is **24.952430 CPU seconds / 185.895536 wall seconds** for at most60 B8 groups/480 episode decisions, including conservative coldsetup and12CPU/120wall native/render/verify/log/monitor allowances. It leaves the separate10CPU audit and10CPU shared reserve. Both180CPU/900wall control gates pass. The90-control fixture's1.5×2400/90 clock projections are7.073000CPU/17.830952wall; they do not force those allowances upward, and are not full-serving measurements or a hard runtime bound. Actual complete controls still require reviewed source, fresh resources/admission and full-trajectory verification.

| Actual guarded stage | Complete tree CPU seconds | Wall seconds |
|---|---:|---:|
| supplied_full_model_qualification | 4.542762 | 8.214058 |
| saved_full_model_qualification_audit | 0.598971 | 0.687701 |

The actual qualification+audit fee is **5.141733CPU**, with8.901759 summed stage-wall and37.682521s receipt envelope. All normal receipts ended within the unchanged300-second window. Nested clocks/forecasts are not added again. At closure release the unused24CPU reservation. The retained allocation becomes **2142.979764/2,400**, leaving **257.020236CPU**. Distinct complete-accounted subset:2583.960560; reported lower bound with the old forced-startup lower bound:2590.565143. Keep the old267CPU reserve as unmeasured, and all failed/engineering costs. Source preparation/closure and external pretraining/overlap/full-host energy/cost are unknown or incompletely metered, not zero.

No new local optimizer update, learned-policy task rollout or closed test was run. The previous1/8 fresh-head result is not a full pretrained-model result. This stage qualifies the interface and provides a cost forecast; **no world-model task-performance advantage is established yet**. The next candidate is one fixed complete eight-start package comparison against the archived direct MDN3/8 and legal RGB+known-physics7/8, with their actual cost scopes preserved in `REFERENCE_COST_SCOPE_REVIEW.md`.
