# One source-specified supplied-checkpoint adaptation

**Yes: the original DINO-WM source supplies a defensible fixed normalization candidate independently of acquiring the full publisher HDF5 action column.** This is a declared adaptation of supplied LeWM weights, not a claim to recover the checkpoint's historical scaler. The missing HF conversion manifest is not an absolute provenance prerequisite for this labeled experiment. No alternative normalizers may be selected using validation/control outcomes.

Root resolved DINO-WM to commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; the saved API metadata has SHA256 `5a8bef548660e86ac5262e808b8640c4e01d67156abafd0e028a65847c65fa75`. This addendum re-read the [loader at that exact commit](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/pusht_dset.py). It defaults to relative actions, reads `rel_actions.pth`, divides by 100, then applies the literal constants below. LeWM's [paper Appendix E](https://arxiv.org/html/2603.19312v1#A5) explicitly identifies the DINO-WM setup and dataset as its PushT source. This provides a scientific rationale for the candidate, beyond a newer environment's default.

The single proposed recipe is:

```
physical command u: two-dimensional relative command
mu = (-0.0087, 0.0068)
sigma = (0.2019, 0.2002)
model action = (u - mu) / sigma
inverse: u = mu + sigma * model_action
```

Apply it componentwise to each chronological raw command, then concatenate five commands into the 10-dimensional block. The literal four-decimal constants define this adaptation; no unknown full-precision values or inferred ddof are substituted. The [fixed DINO environment](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/pusht/pusht_env.py) multiplies the resulting unnormalized command by 100 and, in relative mode, adds current pusher position to form the PD target. That addition belongs inside a qualified relative-action executor. The model/planner receives no future true position. A future absolute-target chunk cannot be converted by borrowing future native pusher positions; retaining the old absolute executor would require a separately specified legal bridge.

There is a real distinction from LeWM's normalization code. Pinned [training](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py) installs a normalizer for each column before splitting. Its [utility](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py) removes NaN-containing rows and uses full-column tensor mean/sample standard deviation. Pinned [evaluation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/eval.py) fits a fresh full-column StandardScaler. Neither path loads these DINO literals as a checkpoint scaler.

An extra normalization is not automatically an incompatibility. In exact arithmetic, recomputing a z-score on the same finite rows eliminates a positive componentwise affine change of the underlying column. Thus storing displacement versus displacement/100, or an already affine-normalized version, can yield the same re-standardized variable. This algebra does not prove equality here: row selection, NaN boundaries, variance convention, rounded constants and float arithmetic remain relevant. The DINO source does not specify the estimator/row set behind its published literals; they are not certified HF moments. No statistics were computed to test this observation.

The strongest supported claim is therefore a **source-specified DINO-statistics adaptation of externally pretrained LeWM**, chosen before results. It can avoid an otherwise unnecessary HF-value acquisition prerequisite. It cannot be labeled exact official LeWM normalization, original-paper reproduction or same-interaction-data training. Qualification must still lock full-model forward/rollout, temporal markers, pixel arithmetic, the relative executor and complete prospective resource coverage. This source-only addendum admits none of those executions and is now stopped.
