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
python main.py --batch_size 64 --epochs 10 --meta_steps 5000 --B 64
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
| `--batch_size` | 64 | Training batch size |
| `--max_length` | 512 | Max token sequence length |
| `--epochs` | 10 | Baseline training epochs |
| `--meta_steps` | 5000 | Meta-training steps |
| `--n_inner_models` | 8 | Population of inner models |
| `--lifetime` | 2000 | Steps before inner model re-init |
| `--T_window` | 2 | Truncated inner loop window |
| `--B` | 64 | Batch size for P_accept formula |
| `--keep_ratio` | 0.7 | Target dataset retention ratio |

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
│   └── filter_stats.json          # Filtering statistics
├── phase5_retrained/
│   ├── retrained_best.pt
│   └── retrained_history.json
└── plots/
    ├── phase1_curves.png          # Baseline training curves
    ├── phase5_curves.png          # Retrain training curves
    ├── score_distribution.png     # DataRater score histogram + CDF
    ├── comparison.png             # MSE/Pearson/FLOPs comparison
    └── mse_overlay.png            # Baseline vs Retrained overlay
```
