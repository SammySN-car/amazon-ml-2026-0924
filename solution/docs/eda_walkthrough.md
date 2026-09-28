# EDA Walkthrough — what each investigation asked, and how the answer emerged

Companion to `../code/v4-0.912/eda.py` (the notebook cells that ran these).
**Scope: EDA-01..EDA-16 only** — the discovery phase before any model or
submission code existed.

How to read: for each investigation — **TARGET** (the question), **RUN**
(what was executed, on what data), **HOW IT EMERGED** (the observation chain),
**DROVE** (the design decision it produced). Samples are seed-0 unless noted.
"pos/neg" = known-true / known-false entity pairs from train_ground_truth.

---

## EDA-01 — Anatomy of a positive pair (sample)
- **TARGET:** Do raw name and address similarities separate true matches from
  random pairs at all?
- **RUN:** Sampled GT pairs + random pairs; quantiles of name fuzz-ratio and
  address jaccard; split both into a name-high/addr-high quadrant grid.
- **HOW IT EMERGED:** pos name-ratio quantiles {25:73, 50:87.5, 75:96.6} vs neg
  {25:27, 50:33.3, 75:39.3} — distributions barely touch above ~56. Exact-name
  matches were only 22% of positives (so exact-only would discard 78%).
  Address jaccard: pos median 0.67 vs neg ≈ 0 — an independent signal.
  Quadrants: 58.7% of pos are name-high/addr-high but 93.7% of neg are
  neither; 3% of random pairs are name-similar *false friends*.
- **DROVE:** Score must combine BOTH signals; exact matching alone is dead.

## EDA-02 — The hard core (sample)
- **TARGET:** Is there a slice of true pairs with *no* usable signal (a
  hard-core that caps achievable recall)?
- **RUN:** Bucketed positives by the two signals; inspected every low/low
  bucket by hand.
- **HOW IT EMERGED:** The "no signal" bucket D came out **0.0%** — every true
  pair has something. The low-name bucket A turned out to be *cross-script
  transliterations* (`Aditya Products` ↔ `आदित्य प्रोडक्ट्स`), not noise:
  76.9% of them share any house number. The name-similar false-friend bucket
  C was suffix boilerplate ("PRIVATE LIMITED") with totally different
  addresses.
- **DROVE:** Address disagreement = near-perfect veto; shared-number features;
  never treat script mismatch as a negative signal.

## EDA-03 — Script, containment, suffix stripping (sample)
- **TARGET:** How do scripts behave; is jaccard the right address metric;
  does suffix stripping pay?
- **RUN:** Char-class stats per country; address containment |a∩b|/min vs
  jaccard quantiles; re-score false friends after stripping legal suffixes.
- **HOW IT EMERGED:** S1 names = 100% Latin; cross-script rate pos 7.2% ≈ neg
  7.3% (script mismatch is a *metric* blind spot, not a signal) — and those
  pairs have addr_jaccard median 0.75, i.e. address decides them. Containment
  ≥0.9 caught **35.7% of pos / 0.0% of neg** where jaccard failed (one address
  is often a literal subset — truncation). Stripping suffixes cut the
  false-friend rate 3.2% → 0.2% (16×).
- **DROVE:** Address containment feature (replaces jaccard as primary);
  suffix-stripped name variants; script-class features.

## EDA-04 — Rule coverage & residual anatomy (sample)
- **TARGET:** How far do three simple rules reach, and what does the residual
  look like?
- **RUN:** Union coverage of R1 (stripped-name ratio ≥60, same-script),
  R2 (addr containment ≥0.9), R3 (shared number); manual read of every miss.
- **HOW IT EMERGED:** U12 = 94.4% pos @ 0.2% neg; U123 = 98.7% @ 2.9% neg.
  The residual 1.3% decomposed into five named failure modes: address
  near-miss, word-order permutation (`Contreras Renewables` ↔ `Renewables,
  Contreras`), domain-form names (`mbarlowcom`), number formatting
  (`002` vs `00002`), word typos. R3's false positives were coincidental
  *small* numbers ("gate 4").
- **DROVE:** The feature list targets those five modes; R3 gated on min digit
  length / address overlap.

## EDA-05 — Structure: duplicates, overlap, key selectivity, match diversity
- **TARGET:** Is an exact-name key usable as a blocking key; are S2 and S3
  copies of each other; how varied are matches within a group?
- **RUN:** Full pool group-by `(country, stripped name)` bucket sizes; sample
  of 3,000 multi-match GT groups.
- **HOW IT EMERGED:** **Found the OLD-normalization bug**: bucket `India|` =
  706,495 rows — the old ASCII-only norm erased every non-Latin name into one
  key (14.9% of pool sat in buckets >50 because of it). Even ignoring the bug,
  exact-name recall was only ~22–30% with a heavy tail (p99 = 624). S2↔S3:
  47% shared exact names but **0.4% shared full records** — independent
  renderings, not copies. Match groups: median 3 distinct *names* per group;
  the inspected groups showed the full variant zoo (transliteration, domains,
  hashtags, word order, numeral substitution, OCR typos, complete renames
  linked only by address).
- **DROVE:** Exact key = one of several, never the only key; kill OLD-norm
  (→ EDA-06/09 validated norm2); blocking must tolerate all variant types.

## EDA-06 — Test-set (France) structure + norm sanity (full data)
- **TARGET:** What does the *test* set look like (unseen country!), and is
  the replacement normalization safe?
- **RUN:** Full test-file statistics; empty-string rates under OLD-norm vs
  norm2 per country; non-ASCII rates; French address/suffix probes.
- **HOW IT EMERGED:** Contract pinned: t1 = **1,732,544** rows (the exact row
  count every submission must emit); countries India 46.8% / US 38.3% /
  France 15.0%. OLD-norm emptied **22.6% of Indian test names** while norm2
  emptied **0.0000% everywhere** — norm2 validated, OLD condemned. France had
  surprises: 5-digit postal codes in only **0.5%** of addresses, French legal
  suffixes (SARL 20%, SAS 14%...), and test S1 itself is 15.7% non-ASCII
  (train S1 was 0% — flagged distribution shift).
- **DROVE:** norm2 is official; ZIP-keyed blocking banned; suffix list
  extended; France-specific handling queued (turned out unnecessary — EDA-16).

## EDA-07 — Field internals: postal keys, house numbers, digits, char inventory
- **TARGET:** Which address primitives are actually present and positionally
  reliable, per country?
- **RUN:** Presence rates of postal codes / digits / leading digits by
  field×country; Unicode category census of every name string.
- **HOW IT EMERGED:** Postal codes dead everywhere (US 10%, India ~0%,
  France 0.5%). House numbers present (91–100% have any digit) but India's
  leading-digit position is only 18–26% (addresses start with locality /
  "H.No") → India needs pattern extraction (H.No, Plot No, #), not position.
  The char census was decisive: US pool = 202,171 non-ASCII names, **100%
  accented Latin** (Spanish-origin) and France likewise → **accent folding is
  required**; India = Devanagari/Kannada/Telugu marks → **never fold those**.
- **DROVE:** No ZIP keys; number features extraction-based; fold policy
  (Latin marks only) — implemented as `fold()`/NFD exact key + norm2.

## EDA-08 — Blocking-key RECALL scorecard (108K-pair sample + 40K neg)
- **TARGET:** Which single keys retrieve positives, at what false rate?
- **RUN:** Per-key hit-rate table on 108K pos + 40K neg, sliced by country
  and script.
- **HOW IT EMERGED:** Exact name only .220; suffix-strip doubled it (.424) at
  zero noise; shared name token .854 (uncapped); **shared numbers = the
  cross-script bridge** (cross-script pos .848 vs name keys ≈ .00);
  unions U1/U2 reached .963/.976. Caveat recorded: token keys were measured
  *uncapped* — cost unknown yet.
- **DROVE:** The key family set (exact, strip, token, number); forced the
  cap-awareness question → EDA-10.

## EDA-09 — Blocking-key SELECTIVITY (full pool, polars, norm2)
- **TARGET:** What do those keys *cost* — bucket sizes and per-S1 candidate
  sums on the full pool?
- **RUN:** Full-pool group-bys with norm2; sum-DF upper bound per S1.
- **HOW IT EMERGED:** With the norm bug gone, max bucket = 1,042 (vs 706,495)
  — confirming EDA-05's diagnosis. But pref4 (4-char prefix) buckets were
  enormous: p90 19,740, est **16 billion pairs → dead as a materialized key**.
  Sum of capped key costs = mean 128.7 candidates/S1 vs a 50 budget, and 63%
  of S1s had *zero* rare name token.
- **DROVE:** Caps and recall must be measured together → EDA-10/11.

## EDA-10 — Cap-aware union recall (GO/NO-GO)
- **TARGET:** Do DF-capped token keys hit the 0.95 recall target affordably?
- **RUN:** Seed-0 sample (105,684 pos + 40K neg); DF-capped unions
  (strict 300/300/100, loose 1000³); anchor checks against EDA-08.
- **HOW IT EMERGED:** **NO-GO**: U_strict 0.734, U_loose 0.853 — both <0.95.
  But negatives under caps were ~0.0000 (caps far too conservative) and the
  *uncapped* union was 0.9994 — so recall was reachable; **selectivity was
  the problem, not coverage**. Also caught a measurement bug: DFs had been
  computed over *distinct name strings* (undercount ×1.26) — all capped
  numbers were re-derived (U_strict corrected 0.7336, U_loose 0.8529).
- **DROVE:** Abandon flat capped single-token keys; hunt mechanisms that
  decouple recall from volume.

## EDA-11 — Cap sweep: recall vs candidate-cost curve
- **TARGET:** Quantify the wall — what does each +0.05 of recall *cost*?
- **RUN:** Caps 100 → uncapped with sum-UB cost accounting and a
  negRate≈sumUB/pool consistency check.
- **HOW IT EMERGED:** 0.95 recall needed cap ~5,000–7,000 → **~5–7K
  candidates/S1 vs a ≤50 budget (~100× gap)**; budget-feasible caps (≤200)
  bought only ~0.60–0.68. Precision was free until cap ≈30K — the wall was
  pure volume (~×5 cost per +0.05–0.09 recall).
- **DROVE:** NO-GO confirmed formally; requirement set: mechanisms that share
  *combinations*, not common singles → EDA-12.

## EDA-12 — Pair-key recall ceilings (shared-token counts)
- **TARGET:** Do ≥2-token / cross-field keys reach 0.95+ where singles died?
- **RUN:** Shared-token count distributions per field; union of pair
  families (N≥2, A≥2, N×A, N×M, A×M) on pos/neg.
- **HOW IT EMERGED:** **U_pair = 0.9948 pos / 0.0817 neg** — and cross-script
  pos ceiling 0.994 (vs 0.62/0.75 under single-token caps): the transliteration
  pairs were carried by rare shared Latin tokens all along. Address pairs
  (A≥2 = .946) were the workhorse.
- **DROVE:** GO on ceiling — but affordability of *common* pair keys was still
  unproven → EDA-13.

## EDA-13 — Budget-greedy pair-key policy (DECISIVE — GO)
- **TARGET:** Can a DF-greedy selection under a hard budget deliver ≥0.95
  recall with ≤~50 candidates/S1?
- **RUN:** Real pool postings restricted to the sample's token universe;
  keys with DF>200 (MAXB) excluded; per S1 sort keys by DF ascending, greedily
  admit until the budget fills; measured recall/mean/p99 per budget on the
  sample.
- **HOW IT EMERGED:** budget 50 → **0.9515 recall @ 44.4 mean cands**;
  budget 100 → 0.9631 @ 89; ablation with singles+exact only @50 = 0.5138 —
  **pair-keys were worth +43.8 points under the same budget**. India trailed
  US by 3.3 points (common-token crowding).
- **DROVE:** The final blocking design (this is what both pipelines run): singles
  + same-field pairs + cross-field pairs + fold-exact, DF-greedy, MAXB=200,
  budget 100 (chosen in EDA-16).

## EDA-14 — Miss autopsy @ budget 50 (key family is NOT the bottleneck)
- **TARGET:** Why the 4.85% misses — bad keys, or not enough budget?
- **RUN:** Decomposed every missed positive pair into: capacity (key
  evaluated, budget full), big_single (shared key DF>200), exact_big,
  no_shared; self-checked totals against EDA-13 (exact replication 0.9515).
- **HOW IT EMERGED:** capacity 68% of misses, big_single 30%, **no_shared
  only 1% (62 pairs)** — no missed match hid in an oversized exact bucket
  (exact_big = 0), anomalies 0. So the key *family* covered everything
  reachable (99.94%); only tuning remained. (Ceiling claim later corrected —
  see EDA-15.)
- **DROVE:** Deprioritize new mechanisms (≤+0.06 pts); tune (MAXB, budget)
  instead.

## EDA-15 — MAXB × budget grid — MAXB is a NO-OP
- **TARGET:** Does raising the per-key DF cap (MAXB) harvest the big_single
  misses, and what is each knob worth?
- **RUN:** Joint grid MAXB {200, 1000, 5000, ∞} × budget {50, 100, 200} with
  keys evaluated once per row; grid(200,50) sanity = 0.9515 exact.
- **HOW IT EMERGED:** **Every MAXB value produced identical tables.**
  Mechanism: budget-greedy can only admit a key costing ≤ remaining budget, so
  raising MAXB above the max budget adds keys greedy can never buy — the
  EDA-14 "big_single" bucket is not harvestable by MAXB at all. This
  *corrected* EDA-14's ceiling: practical ceiling = the budget curve
  (~0.971 @ 200), not 99.94%. Price list produced: 50→100 buys +1.16 pts for
  +44.6 cands; 100→200 buys +0.80 for +89.7.
- **DROVE:** Budget is the only lever; working recommendation 50–100 pending
  test-set reality.

## EDA-16 — Test-set feasibility — France passes; EDA PHASE COMPLETE
- **TARGET:** Does the design transfer to the test set (open-set France),
  what does it cost at scale, and can it run in time?
- **RUN:** 20k/country test sample; test-pool postings; budget projections;
  per-row timing.
- **HOW IT EMERGED:** Sanity matched EDA-06 exactly. France behaved like US
  (means 40.0 @50, French tokens rarer — no pathology, zero% empty at all
  budgets). Real-population means were *below* the pos-sample means (43.1 vs
  44.5 @50 — the sample over-weights multi-match S1s). Volumes: 153M pairs @
  budget 100 ≈ 3.1 GB. Serial full-loop estimate **40.2 h** → the engine must
  be embarrassingly parallel per row.
- **DROVE:** **Budget = 100** (0.9631 recall, 153M pairs); engine design =
  fork-parallel chunks (became `inference.py`, ~10 h on 8 workers); zero% empty is
  *expected* — singletons must be handled by the classifier emitting empty,
  not by blocking.

---

### Cross-reference: EDA → eda.py cells
| EDA | cell in eda.py |
|---|---|
| setup / EDA-01 / EDA-02 (first session) | cells 0–5 |
| EDA-03 | cell 6 |
| EDA-04 | cell 7 |
| EDA-05 | cell 8 |
| EDA-06 | cell 9 |
| EDA-07 | cell 10 |
| EDA-08 | cell 11 |
| EDA-09 | cell 12 |
| EDA-10 / EDA-11 | cells 13–14 |
| EDA-12 | cell 15 |
| EDA-13 (a: postings, b: greedy) | cell 16 |
| blocking-state save / EDA-14 / restore | cells 17–19 |
| EDA-15 | cell 20 |
| EDA-16 | cells 21–22 |

### Meta-lessons the phase produced
1. Three *measurement* bugs were caught and corrected mid-flight (OLD-norm
   emptying 706K names; DF computed on distinct strings undercounting ×1.26;
   MAXB assumed to matter when it was a no-op) — every one was caught by a
   sanity/replication check against a previously measured number.
2. Recall and cost must always be measured **together** (EDA-08 vs EDA-10/11).
3. Negative results drove the design as much as positive ones: every NO-GO
   (flat caps ×2, ZIP keys, pref4, exact-only) closed a wrong path cheaply
   before any production code was written.