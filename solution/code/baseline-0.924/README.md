LB score 0.924 (baseline submission) - full pipeline
=====================================================
Amazon ML Challenge 2026 - Business Entity Resolution.
Public-LB score of this exact code+model: 0.924 (both TSVs, full test set,
1,732,544 Source-1 rows).

WHAT YOU NEED (nothing else - no external state, no pickles, no caches)
----------------------------------------------------------------------
1. This folder.
2. The challenge dataset: dataset/test/test_source{1,2,3}.tsv for inference;
   dataset/train/* additionally only if you retrain.
3. pip install numpy polars rapidfuzz xgboost pandas
   (pandas is needed only by training.ipynb; the pipeline itself is polars/numpy)
   (tested: Python 3.14, numpy 2.5.3, polars 1.41.2, rapidfuzz 3.14.6,
   xgboost 3.4.1; CPU only, ~12 GB RAM, ~5 GB disk for outputs)

PATHS TO EDIT (all hardcoded to /home/ubuntu/... - 5 edit sites)
---------------------------------------------------------------------
inference.py    line ~15:  BASE  = ".../dataset"        -> your dataset folder
           line ~16:  OUTD  = "/home/ubuntu/test_out" -> scratch dir for npz
           line ~189: model  = "/home/ubuntu/xgb_b100.json"   -> where you put the model (not in git)
           line ~190: "/home/ubuntu/threshold_b100.json"      -> ships in this folder; edit the path only
assemble_submission.py line ~12: BASE  = ".../dataset"          -> same dataset folder
           (OUTD/OUTDIR are env vars: OUTD=<npz dir> OUTDIR=output python assemble_submission.py)

FILES
-----
(xgb_b100.json and train_feats_b100.parquet are NOT stored in git - 19 MB / 93 MB;
regenerate them with training.ipynb, or download them from this repository's
GitHub Releases when attached.)
inference.py            Full test pipeline: state build -> token/exact blocking
                     (budget 100, MAXB 200) -> 20 lexical features ->
                     XGBoost score -> thresholded pass.npz per chunk.
                     Self-contained. Chunk-checkpointed, resumable.
assemble_submission.py  Assembles test_out/*.npz into output/matching_results.tsv
                     and output/candidate_pairs.tsv (headers exactly per spec,
                     one row per S1 incl. empty lists).
                     Env: OUTD (npz dir), OUTDIR (output dir),
                     POSTPROC = none | country | country+1to1 (default none).
xgb_b100.json        Trained XGBoost model (20 features, depth 6, lr 0.03,
                     3000 estimators, early stopping, spw ~23).
threshold_b100.json  Emission threshold: 0.97.
train_feats_b100.parquet  Training features (5,707,697 candidate rows x
                     20 features; s1_id/cand_id + features, NO label column -
                     labels are joined from train_ground_truth.tsv).
training.ipynb  Canonical training code: BUILD-1a (blocking on a
                     64,125-S1 train sample -> train_pairs_b100),
                     BUILD-1b (features -> train_feats_b100.parquet),
                     BUILD-1c-v2 (GT labels + train/val split = sort unique
                     s1_id, shuffle rng(0), 80/20 + XGB train + threshold
                     sweep -> xgb_b100.json + threshold_b100.json).
                     Also contains the EDA that motivated the design.
validate_submission.py  Official validator (stdlib only).

REPRODUCE THE SUBMISSION FILES FROM TEST DATA
---------------------------------------------
  python inference.py                       # ~10 h on 8 cores -> <OUTD>/*.npz
  OUTD=<the OUTD you set above> OUTDIR=output python assemble_submission.py
                                               # -> output/*.tsv
  python validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir <your dataset>/test
  # -> PASS. Portal upload during the challenge: both TSVs.

RETRAIN
-------
  Run notebook cells BUILD-1a-FAST/1b (blocking + FE, ~25 min) to rebuild
  train_feats_b100.parquet (absent from git - fetch from Releases if
  attached), then BUILD-1c-v2 for labels/split/train -> model + threshold.

METHOD IN ONE PARAGRAPH
-----------------------
Blocking: NFD accent-fold exact key + name/address/2+-digit token postings
restricted to the query token universe + pair-key intersections (df <= 200),
greedy by ascending df into a budget of 100 candidates per S1 (blocking
recall 96.31% on validation). Scoring: 20 lexical features (name/address
ratio, exact/suffix/fold/prefix matches, token containment/Jaccard, address
containment/missing, shared numbers, script class, length diff, country
equal) -> XGBoost, emit pairs >= 0.97.