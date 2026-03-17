# P8 Inner Reset Suite Summary

Date: 2026-03-17

Goal: compare three Phase 2 inner-model reset strategies under the same simple/original DataRater setup:

1. `random_init`
2. `carryover`
3. `checkpoint_bank` with bank schedule `5,5,6,6,7,7,8,8`

Common setup:

- `datarater_arch=single`
- `outer_sampling=random`
- `data_mode=all`
- `epochs=10`
- `meta_steps=1500`
- `retrain_epochs=10`
- `batch_size=32`
- `meta_batch_size=16`
- `T_window=8`
- `T_backprop=8`
- `n_inner_models=8`
- `keep_ratio=0.95`
- `outer_objective=pearson`
- random control: `matched_source_counts`

## Results

| Strategy | Baseline MSE | DataRater Retrained MSE | Matched-Source Random MSE | Delta vs Baseline | Delta vs Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_init` | 0.6702 | 0.7178 | 0.6777 | +0.0475 | +0.0400 |
| `carryover` | 0.6489 | 0.6724 | 0.6525 | +0.0235 | +0.0199 |
| `checkpoint_bank` | 0.6717 | 0.7256 | 0.6765 | +0.0539 | +0.0491 |

## Run Paths

- `random_init`
  - `/home/ubuntu/YutongWorkstation/DataRater/experiments/p8_single_randominit_bs32_metabs16_tau0.5_B32_outer-pearson_T8_Tb8_n8_ms1500_seed42/p4_mixflow_single_randomouter_20260316_014422/results.json`
- `carryover`
  - `/home/ubuntu/YutongWorkstation/DataRater/experiments/p8_single_carryover_bs32_metabs16_tau0.5_B32_outer-pearson_T8_Tb8_n8_ms1500_seed42/p4_mixflow_single_randomouter_20260316_113716/results.json`
- `checkpoint_bank`
  - `/home/ubuntu/YutongWorkstation/DataRater/experiments/p8_single_bankmix_bs32_metabs16_tau0.5_B32_outer-pearson_T8_Tb8_n8_ms1500_seed42/p4_mixflow_single_randomouter_20260316_212821/results.json`
- bank manifest
  - `/home/ubuntu/YutongWorkstation/DataRater/experiments/p8_inner_bank_mix_seed42/inner_init_bank_20260316_010423/manifest.json`

## Interpretation

- `random_init` is clearly bad in this setting.
- `carryover` partially fixes the problem:
  - it improves substantially over `random_init`
  - but it still loses to both full-data baseline and matched-source random
- `checkpoint_bank` did not help here:
  - it was worse than `carryover`
  - and slightly worse than plain `random_init`

## Conclusion

For this simple/original DataRater setup, the user's hypothesis was directionally right that repeated random resets are harmful, but the tested bank warm-start implementation did not recover the expected benefit.

Current ranking:

1. `carryover` (best of the three, but still negative)
2. `random_init`
3. `checkpoint_bank`

Practical decision:

- keep the summary as a negative/partial result
- do not continue iterating on this exact `checkpoint_bank` design right now
- switch back to the teacher-mainline experiments
