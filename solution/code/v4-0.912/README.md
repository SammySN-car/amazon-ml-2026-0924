# v4 pipeline - public LB 0.912 (36-feature LightGBM)

Entity matching between a business-description source (S1) and two entity
catalogues (S2, S3). The pipeline emits `output/matching_results.tsv`
(emitted matches per S1) and `output/candidate_pairs.tsv` (blocking
candidates per S1) for all 1,732,544 test S1 rows.

Scoring model: gradient-boosted trees over 36 lexical features
(name/address similarity + numeric + country + script-class), trained on a
seeded sample of the training data with candidates produced by the same
blocking used at inference. Emission threshold is selected by a deep sweep
on an honest held-out group split; the selected global threshold is 0.965
and is stored in `v4_model.json`.

## Environment

- Linux x86_64, CPU only (tested: 8 vCPU / 63 GB RAM), CPython 3.14
- `pip install -r requirements.txt`
- Peak RAM ~12 GB (state build + 8 fork workers); disk ~40 GB free
- Paths are constants at the top of each script, all defaulting to
  `/home/ubuntu/...` - see **Paths to edit** below.

## Paths to edit (defaults are all `/home/ubuntu/...`)

- `BASE` - dataset root (contains `train/` and `test/`): top of
  `prepare_data.py`, `train.py`, `inference.py`, `assemble_submission.py`.
- `inference.py` - `OUTD` (checkpoint dir) and `MODEL` (model spec path);
  both also readable as env vars (`OUTD`, `MODEL`), which is what the
  commands below use.
- `train.py` - `GED` (feature-cache dir) plus the save paths of
  `v4_model.json`, `v4_xgb20.json`, `v4_xgb.json`, `v4_lgbm.txt`,
  `v4_cat.cbm` and `train_feats_b100_v4.parquet`. Input/output of the
  candidate file is env-controlled: `CANDS` (read) and `CANDS_OUT` (write,
  set in `prepare_data.py`).
- `rescore.py` and `assemble_submission.py` are env-var only - no edits.
  If your data lives elsewhere, either edit `BASE` or create
  `/home/ubuntu/dataset -> <your dataset dir>` (the scripts do not search
  relative paths).

## Run order (from this directory)

```
# 1. Blocking candidates for training (seeded sample; ~20 min, 8 workers)
python prepare_data.py            # -> train_cands.parquet

# 2. Feature build + model training + threshold sweep (~45 min)
python train.py                   # -> v4_model.json + v4_xgb20.json / v4_xgb.json /
                                   #    v4_lgbm.txt / v4_cat.cbm (paths: see above)
                                   # prints validation macro F0.5 per model arm + chosen threshold

# 3. Test pipeline: blocking -> features -> score (~10 h on 8 cores, resumable)
OUTD=test_out_v4 python inference.py  # -> test_out_v4/c*.cand.npz, c*.pass.npz,
                                   #    c*.X.npy (also set MODEL if you edited its path)

# 4. Assemble submission TSVs (~1 min)
OUTD=test_out_v4 OUTDIR=output python assemble_submission.py
                                   # -> output/matching_results.tsv
                                   #    output/candidate_pairs.tsv

# 5. (Optional) iterate on threshold or model WITHOUT re-running step 3:
#    re-predicts from the saved feature matrices, minutes instead of ~10 h
SRCD=test_out_v4 OUTD=out_rescore python rescore.py
OUTD=out_rescore OUTDIR=output2 python assemble_submission.py
                                   # TAU defaults to the spec threshold (0.965);
                                   # POSTPROC=none|country|country+1to1 on assemble_submission
```

`inference.py` is chunk-checkpointed and safe to re-run; it consumes
`v4_model.json` (`feats`, `models[{kind,path,weight}]`, `threshold`).

`train.py` reads `train_cands.parquet` (step 1) if present, otherwise falls
back to `train_feats_b100.parquet` - the baseline's precomputed feature file
(only its `s1_id`/`cand_id` columns are read; labels, text joins and features
are recomputed inside the script from raw TSVs). That file is not in git:
see `../baseline-0.924/README.md` for how to obtain or regenerate it.

`assemble_submission.py` env vars: `OUTD` (npz dir), `OUTDIR` (output dir),
`POSTPROC` = `none | country | country+1to1` (matching-results postprocessing;
country filter and 1-to-1 resolution were measured neutral on validation and
default to `none`).

## Feature set (`features.py`)

36 features: the original 20 (name ratio, exact/suffix/fold/prefix matches,
token containment/Jaccard, address containment/Jaccard/exact/missing,
shared 2/3-digit numbers, both-numeric, script class A/B/diff, length diff,
country equal) plus 16 extensions (`features.py` names in brackets):
name Jaro-Winkler [n_jw], name partial / token-sort / token-set ratios
[n_partial, n_token_sort, n_token_set], address Jaro-Winkler + token-sort /
partial ratios [a_jw, a_token_sort, a_partial], name bigram / trigram
Jaccard [n_char2_jacc, n_char3_jacc], first-token equality [first_tok_eq],
address prefix-2 / prefix-5 equality [addr_pref2_eq, addr_pref5_eq],
harmonic and min combinations of (name ratio, address containment)
[harm_nr_aj, min_nr_aj], log-scale first-number similarity [lognum_sim],
address length diff [a_lendiff]).

## Blocking (same rules in `prepare_data.py` and `inference.py`, one exception)

1. Normalize names/addresses (uppercase, strip ASCII punctuation, collapse
   whitespace); accent-fold (NFD, drop combining) for an exact key.
2. Index pool (S2+S3) into name-token / address-token / 2+-digit-number
   postings restricted to the query token universe, plus fold-exact buckets.
3. Per S1: single-key postings (sorted by document frequency) + pairwise
   key intersections (df <= 200), then greedy selection by ascending df into
   a budget of 100 candidates.

Exception: at test time `inference.py` also restricts candidates to the S1's
own country (cross-country matches are 0 by design). Training candidates from
`prepare_data.py` are not country-partitioned - the shipped model was trained
on that asymmetry, so the code reproduces it as-is.

Blocking recall on validation: 96.31% at budget 100.