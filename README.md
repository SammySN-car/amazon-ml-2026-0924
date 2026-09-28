# Amazon ML Challenge 2026 - Business Entity Resolution (independent line, LB 0.924 / 0.912)

An independent, CPU-only solution for the Business Entity Resolution challenge.
This repository contains two end-to-end pipelines (a 20-feature XGBoost baseline
and a 36-feature LightGBM v4) plus methodology and EDA documentation, all
validated end-to-end against the official validator.

## Results

| System   | Features | Model    | Validation macro-F0.5 | Public LB |
|----------|----------|----------|-----------------------|-----------|
| Baseline |       20 | XGBoost  | 0.9082                | **0.924** |
| v4       |       36 | LightGBM | 0.9356                | **0.912** |

The two leaderboard scores deviate from validation in opposite directions
(baseline +0.016, v4 -0.024): validation contains no French rows, France is
unseen in training, and France carries 15% of leaderboard weight. Details are in
`solution/docs/methodology.md` (section 5).

## Repository layout

```text
solution/
  code/baseline-0.924/          0.924 submission (20 features, XGBoost)
    inference.py                test pipeline: blocking -> features -> score -> npz
    assemble_submission.py      npz chunks -> output/*.tsv
    validate_submission.py      official validator (stdlib only)
    training.ipynb              BUILD-1a/1b/1c-v2 training flow + EDA
    threshold_b100.json         emission threshold (0.97)
    README.md                   requirements, paths to edit, reproduce, retraining
  code/v4-0.912/                0.912 pipeline (36 features, LightGBM)
    prepare_data.py             training blocking -> train_cands.parquet
    train.py                    features + training + threshold sweep
    inference.py                test pipeline (chunk-checkpointed)
    assemble_submission.py      npz chunks -> output/*.tsv
    rescore.py                  threshold sweeps without re-running inference
    features.py                 36 feature definitions
    eda.py                      analysis for the 16-step EDA
    requirements.txt            pinned environment
    README.md                   requirements, paths to edit, reproduce, feature set, blocking
  docs/
    methodology.md              problem analysis, blocking, model, results, error analysis
    eda_walkthrough.md          the 16-step EDA that drove every design decision
    amazon_ml_challenge_2026_problem_statement.pdf   official challenge rules
    amazon_ml_challenge_2026_guidelines_and_key_instructions.pdf   official submission guidelines
```

## Reproducing

Each code folder has its own README with the exact commands, environment and
expected runtimes. Everything is CPU only; a full test-set run takes about 10
hours on 8 cores.

## Large artifacts

`xgb_b100.json` (19 MB) and `train_feats_b100.parquet` (93 MB) are not stored
in git; regenerate them with `solution/code/baseline-0.924/training.ipynb`, or
download them from this repository's GitHub Releases when attached. The
challenge dataset is the organizers' data and is not included.

## Team

- Part of Team TheAnarchy.
- Final team pipeline (LB 0.968): https://github.com/RavishCRZ27/amazon-ml-challenge-2026
- Samar Nathani (@SammySN-car) - Team TheAnarchy
