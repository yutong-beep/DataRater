# DataRater: Meta-Learned Dataset Curation for Binding Affinity

Meta-learning framework that assigns quality scores to training data using bilevel optimization, applied to protein-protein binding affinity prediction.

## Project Structure

```
.
├── baseline_trainer.py   # Phase 1 & 5: standard supervised training
├── data_utils.py         # Download, tokenize, DataLoader factory
├── main.py               # Single entry point — runs all 5 phases
├── meta_trainer.py       # Phase 2: meta-training wrapper with logging
├── model.py              # Core models & bilevel meta-learning logic
├── scoring.py            # Phase 3+4: scoring & filtering wrapper
├── viz.py                # Visualization: curves, distributions, comparisons
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (all 5 phases)
python main.py

# With custom settings
python main.py --batch_size 64 --meta_batch_size 16 --epochs 10 --meta_steps 5000 --B 64 --temperature 0.5

# Run with all source files (excluding Combined_train to avoid duplication)
python main.py --data_mode all --output_dir experiments/mode_all

# Run phase 2-5 with scale-robust outer objective
python main.py --phase 2,3,4,5 --outer_objective mse_norm --data_mode all

# Run phase 3-5 and include random baseline retrain in phase 5
python main.py --phase 3,4,5 --datarater_ckpt path/to/datarater.pt --random_baseline --random_mode matched_source_counts --random_seed 42

# Run random-only retrain for an existing run directory
python main.py --random_only --run_dir experiments/run_YYYYMMDD_HHMMSS --random_mode matched_source_counts --random_seed 42
```

## Running Individual Phases

```bash
# Phase 1 only: Baseline training
python main.py --phase 1

# Phase 1+2: Baseline + Meta-training
python main.py --phase 1,2

# Phase 3+4+5 with pre-trained DataRater
python main.py --phase 3,4,5 --datarater_ckpt path/to/datarater.pt
```

## Task Requirements Mapping

| Task | Implementation | Flag |
|------|---------------|------|
| a. Periodic re-init | `train_datarater()` offset-based lifetime | Always on |
| b. Outer-only optimizer.step() | Functional inner updates | Always on |
| c. First-order ablation | `--ablation` | `use_first_order_ablation` |
| d. Within-batch softmax | `F.softmax(scores/tau, dim=0)` | Always on |
| d. CDF-based P_accept | `filter_dataset()` + `scoring.py` | Phase 3+4 |
| e. Truncated window T=2 | `--T_window 2` (default) | Configurable |
| f. Sample 1 inner model | `--sample_one_inner` | `sample_one_inner` |
| g. Retrain & compare | Phase 5 + `viz.py` comparison | Phase 5 |

## Ablation Study

```bash
# Full backprop through unrolled states (default)
python main.py

# First-order ablation: backprop only through last state
python main.py --ablation

# Sample 1 inner model instead of all 8
python main.py --sample_one_inner

# Both ablations combined
python main.py --ablation --sample_one_inner
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `experiments` | Root directory for run outputs |
| `--data_mode` | `combined_train` | `combined_train` or `all` (all non-Combined parquet files) |
| `--batch_size` | 64 | Training batch size |
| `--meta_batch_size` | 16 | Phase-2 meta-training batch size (separate from baseline/retrain batch size) |
| `--max_length` | 512 | Max token sequence length |
| `--epochs` | 10 | Baseline training epochs |
| `--meta_steps` | 5000 | Meta-training steps |
| `--n_inner_models` | 8 | Population of inner models |
| `--lifetime` | 2000 | Steps before inner model re-init |
| `--T_window` | 2 | Truncated inner loop window |
| `--temperature` | 0.5 | Softmax temperature `tau` for DataRater weighting |
| `--outer_objective` | `mse_norm` | Outer loss type: `mse_norm`, `pearson`, `cosine`, or `mix` |
| `--alpha` | 0.5 | Mixing weight for `mix` objective: `alpha*(1-pearson)+(1-alpha)*mse_norm` |
| `--outer_eps` | 1e-8 | Epsilon for Pearson/Cosine stability |
| `--mse_norm_eps` | 1e-6 | Epsilon for source-normalized MSE denominator |
| `--B` | 64 | Batch size for P_accept formula |
| `--keep_ratio` | 0.7 | Target dataset retention ratio |
| `--retrain_epochs` | 10 | Epochs for retraining on filtered data |
| `--random_baseline` | `False` | Also run random baseline retrain in Phase 5 |
| `--random_seed` | 42 | Seed used for random subset sampling |
| `--random_mode` | `matched_source_counts` | Random baseline strategy: `matched_source_counts`, `stratified_ratio`, `uniform` |
| `--random_only` | `False` | Run only random baseline retrain for an existing run dir, then exit |
| `--run_dir` | `None` | Existing run directory used by `--random_only` |

## Random Baseline (`--random*`)

- Integrated mode: add `--random_baseline` in normal pipeline runs; when Phase 5 executes, it retrains both DataRater subset and matched random subset (same `keep_act`).
- Random-only mode: `--random_only --run_dir <existing_run>` loads prior run artifacts and executes only random baseline retrain.
- Supported random subset strategies (`--random_mode`): `matched_source_counts` (default, match kept source counts), `stratified_ratio` (per-source ratio with exact-size adjustment), `uniform` (global random sample).

## Outer Objectives (Phase 2)

- `mse_norm` (default): source-normalized MSE using source std from **train split only**
- `pearson`: outer loss `1 - rho(pred, target)` (scale-invariant)
- `cosine`: outer loss `1 - cosine(pred, target)` (mean-centered)
- `mix`: `alpha*(1-rho) + (1-alpha)*mse_norm`

### Experiment Commands

```bash
# 1) Source-normalized MSE
python main.py --phase 2,3,4,5 --outer_objective mse_norm

# 2) Pearson
python main.py --phase 2,3,4,5 --outer_objective pearson

# 3) Mixed objective
python main.py --phase 2,3,4,5 --outer_objective mix --alpha 0.5
```

## Data Modes

- `combined_train` (default): uses only `Combined_train.parquet`
- `all`: uses all whitelisted source files **except** `Combined_train.parquet`, with schema normalization in `data_utils.py`

## Output Structure

```
experiments/run_YYYYMMDD_HHMMSS/
├── config.json                    # Run configuration
├── pipeline.log                   # Full log
├── results.json                   # All metrics & results
├── phase1_baseline/
│   ├── baseline_best.pt           # Best checkpoint
│   ├── baseline_final.pt          # Final checkpoint
│   └── baseline_history.json      # Per-epoch metrics
├── phase2_datarater/
│   ├── datarater.pt               # Trained DataRater
│   └── meta_config.json
├── phase34_scoring/
│   ├── all_scores.npy             # Raw scores for all points
│   ├── all_scores_with_data.jsonl # Score + mapped raw sample fields per row
│   ├── filter_stats.json          # Filtering statistics (includes kept_source_proportions)
│   ├── random_kept_indices.npy    # Random baseline subset indices (if --random_baseline)
│   └── random_info.json           # Random subset metadata (mode/seed/keep_act, etc.)
├── phase5_retrained/
│   ├── retrained_best.pt
│   └── retrained_history.json
├── phase5_random/                 # Present when --random_baseline or --random_only is used
│   ├── random_best.pt
│   ├── random_history.json
│   ├── random_kept_indices.npy
│   ├── random_info.json
│   └── random_results.json
└── plots/
    ├── phase1_curves.png          # Baseline training curves
    ├── phase5_curves.png          # Retrain training curves
    ├── score_distribution.png     # DataRater score histogram + CDF
    ├── comparison.png             # MSE/Pearson/FLOPs comparison
    └── mse_overlay.png            # Baseline vs Retrained overlay
```
