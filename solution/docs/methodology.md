# Amazon ML Challenge 2026 - Business Entity Resolution: Independent Line

**Team Name:** TheAnarchy

**Author:** Samar Nathani ([@SammySN-car](https://github.com/SammySN-car)) - Team TheAnarchy

**Submission Date:** 2026-09-27

**Final team pipeline (LB 0.968):** https://github.com/RavishCRZ27/amazon-ml-challenge-2026

---

## 1. Executive Summary
A CPU-only blocking + gradient-boosting pipeline: pair-key blocking with a
DF-greedy budget (153M candidates, 96.3% train recall) feeds a 36-feature
LightGBM matcher whose threshold is tuned for macro F_0.5 on a group-held-out
validation set. Key innovations: cross-field *pair-keys* selected greedily by
document frequency under a hard per-entity budget (vs. flat token caps, which
were proven infeasible), and address-containment features that survive the
address truncation common in both sources.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from a 16-step EDA (see `eda_walkthrough.md`):

- **Name similarity alone is a trap:** 3% of random pairs are name-similar
  false friends (legal-suffix boilerplate); exact names cover only 22% of
  true matches.
- **Address truncation breaks jaccard; containment replaces it:** one address
  is often a literal subset of the other - address *containment* >=0.9 catches
  35.7% of true pairs at 0.0% false-positive rate.
- **Script and accent handling:** reference entities are pure ASCII, but
  candidate pools contain Devanagari/Kannada/Telugu and accented Latin. Latin
  marks are accent-folded; non-Latin scripts are never folded - cross-script
  pairs are decided by shared numbers and address overlap (name similarity is
  identically 0 across scripts).
- **Postal codes are dead as keys** (present in only ~10% US / 0.5% France /
  ~0% India addresses) - blocking must not depend on them.
- **The test set adds an unseen country (France, 15%)** whose addresses carry
  effectively no ZIP codes; the design was validated to transfer (France
  candidates track US rates, no pathology).
- **Exact-name keys are low-recall (22%)** with a heavy tail (one key bucket
  held 706k rows due to an early normalization bug, fixed and re-validated).

### 2.2 Solution Strategy
Blocking -> feature extraction -> LightGBM -> threshold tuned on macro
(group-held-out) F_0.5 -> post-processing options -> TSV emission with explicit empty rows for
singletons (5.6% of entities are true singletons; emitting nothing scores 1.0).

**Approach:** Blocking + classifier.

**Core innovation:** Pair-key blocking (share >=2 name/addr/number tokens or
a cross-field token combination) selected per entity by DF-greedy admission
under a hard budget - measured +43.8 recall points over single-token keys at
the same budget.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** suffix-stripped exact name; rare name/address/number
  tokens (DF <= 200, the MAXB cap); same-field token pairs; cross-field pairs
  (name x addr, name x number, addr x number); accent-folded exact key.
- **Selection policy:** keys sorted by DF ascending, admitted greedily while
  the union stays within **budget 100** candidates per S1.
- **Candidate pairs generated:** 153,356,491 (test), mean ~89 per S1.
- **Recall guarantees:**
  - Measured end-to-end retrieval on held-out ground truth: **0.9631 recall**
    at budget 100 (0.9515 at budget 50).
  - A miss autopsy showed 68% of residual misses are pure budget-capacity and
    1% have zero token overlap - the key family itself reaches 99.9%.
  - Numbers act as the cross-script bridge (84.8% of cross-script pairs share
    a number) so transliteration pairs survive.

---

## 4. Matching Model

**Features used (36):**
- Name features: fuzz ratio (raw + suffix-stripped), token containment /
  Jaccard, token-sort/partial ratios, exact-match flags, script-class
  agreement, shared-token counts.
- Address features: containment |a∩b|/min (primary), jaccard, token overlap,
  missing-address flags, shared-number features (any / >=2-digit /
  digit-run overlap).
- Other: country agreement, length deltas, number-format variants
  (zero-padding tolerance), cross-script indicator.

**Model type:** LightGBM (single model, selected over XGBoost/CatBoost and
over 2-model ensembles in a controlled gate: val macro 0.9356 vs 0.9323 xgb /
0.9327 best ensemble).

**Threshold selection method:** per-threshold macro F_0.5 on a group-held-out
validation split (entity-disjoint via sorted-id + seeded shuffle), best tau =
0.965; re-sweepable post-hoc from saved feature matrices (predict-only).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation 0.9356 (gate split, 12,090 groups);
  public LB **0.912** (v4, submitted 2026-09-27). The earlier baseline system (20 features,
  XGBoost, threshold 0.97) scored validation 0.9082 and public LB **0.924**. The two
  leaderboard values deviate from validation in opposite directions (+0.016 vs -0.024)
  because validation contains no French rows: France is unseen in training but carries
  15% of leaderboard weight, so the tuned v4 selection moved away from it.
- **Common false positives (wrong merges):** name-similar entities with
  boilerplate suffixes ("PRIVATE LIMITED", "LLC") but different addresses -
  concentrated just under the threshold; also duplicate entity IDs across
  countries. An address-disagreement veto and country filter were tested:
  cross-country matches are already 0 and closure-style grouping hurts
  (measured), so precision is handled by the threshold itself.
- **Common false negatives (missed matches):** budget-bound capacity misses
  (the blocking budget, not the key set), India common-key crowding (-3.3
  recall pts vs US), cross-script pairs where address overlap is partial, and
  complete renames that share only an address.

---

## 6. Conclusion
A deliberately measured pipeline: every design choice (containment over
jaccard, pair-keys over flat caps, budget 100, LightGBM, tau 0.965) is backed
by a documented experiment, and three measurement bugs were caught and
corrected by replication checks before shipping. The main lesson: in entity
resolution with truncated addresses, *recall is a blocking-economics problem
and precision is a threshold problem* - treat them separately.

---

## Appendix

### A. Code Artefacts
Code lives in `solution/code/v4-0.912/` (this system) and
`solution/code/baseline-0.924/` (the 0.924 baseline). Entry points to
reproduce the two TSVs (CPU, dataset dir as only input; run from
`solution/code/v4-0.912/` after editing the data paths at the top of each
script - see that folder's README):

1. `python prepare_data.py`    - sampled training candidates (seed 0)
2. `python train.py`           - feature build + model/threshold gate -> v4_model.json
3. `OUTD=test_out_v4 python inference.py` - test blocking + features -> test_out_v4/c*.cand.npz / c*.pass.npz / c*.X.npy
4. `OUTD=test_out_v4 OUTDIR=output python assemble_submission.py` - merges checkpoints -> output/matching_results.tsv + candidate_pairs.tsv
5. (Optional) `rescore.py`     - re-threshold/re-model from the saved `X.npy`
   matrices (minutes; no re-blocking), then re-run step 4.

Validation: `output/*.tsv` were checked with the official validator,
`validate_submission.py` (shipped unmodified in
`solution/code/baseline-0.924/`), before every upload - PASS ("Safe to
submit") was required.

`features.py` is the shared feature core (identical train/test code);
`rescore.py` enables threshold/model iteration without re-running
inference; `eda.py` ships the exploratory analysis.

### B. Additional Results
- Blocking recall budget curve: 50 -> 0.9515, 100 -> 0.9631, 200 -> 0.9711.
- Model gate (val macro F_0.5): xgb-36 0.9323, lgbm-36 **0.9356**, cat-36
  0.9267, ensembles 0.9323-0.9327.
- Post-processing gate: country filter no-op (0 cross-country emissions),
  1-to-1 resolver no-op, group closure harmful (0.867-0.904) - reported
  honestly; threshold-only emission was kept.

---
