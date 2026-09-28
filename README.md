# Amazon ML Challenge 2026 - Business Entity Resolution: Independent Line

An independent, CPU-only solution for the Business Entity Resolution challenge.
Two systems, both verified end-to-end against the official validator:

| System | Validation F0.5 | Public LB |
|---|---|---|
| Baseline (20 features, XGBoost, threshold 0.97) | 0.9082 | **0.924** |
| v4 (36 features, LightGBM, threshold 0.965) | 0.9356 | **0.912** |

The two leaderboard scores deviate from validation in opposite directions (+0.016
vs -0.024): validation contains no French rows, France is unseen in training but
carries 15% of leaderboard weight. Details in `solution/docs/methodology.md` (section 5).

Part of Team TheAnarchy. Final team pipeline (LB 0.968):
https://github.com/RavishCRZ27/amazon-ml-challenge-2026

## Repository layout

```
solution/
  code/baseline-0.924/      0.924 submission: inference, assemble, training notebook
  code/v4-0.912/            v4 pipeline: prepare_data -> train -> inference -> assemble
                            (+ rescore for threshold sweeps, eda for analysis)
  docs/methodology.md       problem analysis, blocking, model, results, error analysis
  docs/eda_walkthrough.md   the 16-step EDA that drove every design decision
  docs/amazon_ml_challenge_2026_problem_statement.pdf    official challenge rules
  docs/amazon_ml_challenge_2026_guidelines_and_key_instructions.pdf
```

## Reproduce

Each code folder has its own README with exact commands, environment and
expected runtimes (CPU only; full test set ~10 h on 8 cores).

## Large artifacts

`xgb_b100.json` (19 MB) and `train_feats_b100.parquet` (93 MB) are not stored in
git - regenerate them via `solution/code/baseline-0.924/training.ipynb`, or
download them from this repository's GitHub Releases when attached.
The challenge dataset is the organizers' data and is not included.