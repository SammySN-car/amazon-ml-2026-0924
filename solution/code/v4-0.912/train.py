#!/usr/bin/env python3
# train.py - trains and gates the v4 model (uses all cores).
# Rebuilds train candidate features with features (36), labels from FULL GT,
# trains xgb/lgbm/cat variants, threshold sweep, end-to-end validation (baseline anchor F0.5 0.9082).
# Writes /home/ubuntu/v4_model.json consumed by inference.py.
# Regression check: fresh XGBoost on the 20 baseline features at t=0.97 must score ~0.9082 val macro F0.5.
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import gc, json, time, sys
import numpy as np
import polars as pl
import features
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from multiprocessing import get_context

BASE = "/home/ubuntu/dataset/student_resource/dataset"
GED = "/home/ubuntu/gate_fe"
os.makedirs(GED, exist_ok=True)
for f in os.listdir(GED):                          # keep completed .npz for resume; wipe partials
    if f.endswith(".tmp"): os.remove(os.path.join(GED, f))
V4 = features.FEATS
NF = features.NF

def pick(cols, *pats):
    for p in pats:
        for c in cols:
            if p.lower() in c.lower(): return c
    raise ValueError(f"{pats} not in {cols}")

t0 = time.time()

# ---- rows (ids) + raw text joins ----
CANDS = os.environ.get("CANDS", "/home/ubuntu/train_cands.parquet")
SRC = CANDS if os.path.exists(CANDS) else "/home/ubuntu/train_feats_b100.parquet"
print(f"candidates: {SRC}", flush=True)
f_old = pl.read_parquet(SRC).select(["s1_id", "cand_id"])
N = f_old.height
print(f"train candidate rows: {N:,}", flush=True)

s1f = pl.read_csv(f"{BASE}/train/train_source1.tsv", separator="\t")
i1 = pick(s1f.columns, "entity_id", "id")
n1 = pick(s1f.columns, "business_name", "name")
a1 = pick(s1f.columns, "business_address", "address")
c1 = pick(s1f.columns, "country")
s1sel = s1f.select([pl.col(i1).alias("s1_id"), pl.col(n1).alias("n1"), pl.col(a1).alias("a1"), pl.col(c1).alias("ct1")])
del s1f
e2f = pl.read_csv(f"{BASE}/train/train_source2.tsv", separator="\t")
e3f = pl.read_csv(f"{BASE}/train/train_source3.tsv", separator="\t")
def slim(df):
    i = pick(df.columns, "entity_id", "id")
    n = pick(df.columns, "business_name", "name")
    a = pick(df.columns, "business_address", "address")
    c = pick(df.columns, "country")
    return df.select([pl.col(i).alias("eid"), pl.col(n).alias("n2"), pl.col(a).alias("a2"), pl.col(c).alias("ct2")])
e23 = pl.concat([slim(e2f), slim(e3f)])
del e2f, e3f
f2 = f_old.join(s1sel, on="s1_id").join(e23, left_on="cand_id", right_on="eid").drop("eid", strict=False)  # polars drops right key
assert f2.height == N, (f2.height, N)
print(f"[{(time.time()-t0)/60:.1f} min] joins done", flush=True)

# ---- labels from FULL GT (findings rule) ----
gt = pl.read_csv(f"{BASE}/train/train_ground_truth.tsv", separator="\t")
ic = pick(gt.columns, "source1", "entity_id")
im = pick(gt.columns, "matched", "ids")
gt_pairs = (gt.select([pl.col(ic).alias("s1_id"), pl.col(im).fill_null("").alias("m")])
              .select(["s1_id", pl.col("m").str.split(",").alias("m")])
              .explode("m")
              .select(["s1_id", pl.col("m").str.strip_chars().alias("cand_id")])
              .filter(pl.col("cand_id").str.len_chars() > 0)
              .with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
              .unique())
del gt
f2 = f2.join(gt_pairs, on=["s1_id", "cand_id"], how="left").with_columns(
    pl.col("y").fill_null(0))
del gt_pairs
gc.collect()
print(f"labels: pos {int(f2['y'].sum()):,} ({f2['y'].mean():.4f})", flush=True)

# ---- FE via fork workers (numpy npz only in children) ----
NM1 = f2["n1"].fill_null("").to_numpy()
AD1 = f2["a1"].fill_null("").to_numpy()
CT1 = f2["ct1"].fill_null("").to_numpy()
NM2 = f2["n2"].fill_null("").to_numpy()
AD2 = f2["a2"].fill_null("").to_numpy()
CT2 = f2["ct2"].fill_null("").to_numpy()
s1_arr = f2["s1_id"].to_numpy()
cand_arr = f2["cand_id"].to_numpy()
y = f2["y"].to_numpy().astype(np.int8)
del f2
gc.collect()

SL = 100_000
TK = [(a, min(a + SL, N)) for a in range(0, N, SL)]
todo = [c for c in TK if not os.path.exists(f"{GED}/c{c[0]:06d}.npz")]
print(f"[{(time.time()-t0)/60:.1f} min] FE slices {len(TK)} | todo {len(todo)}", flush=True)

def fe_work(ck):
    a, b = ck
    try:
        X = np.empty((b - a, NF), dtype=np.float32)
        for i in range(a, b):
            X[i - a] = features.featurize(NM1[i], NM2[i], AD1[i], AD2[i], CT1[i], CT2[i])
        with open(f"{GED}/c{a:06d}.tmp", "wb") as fh:
            np.savez(fh, X=X)
        os.replace(f"{GED}/c{a:06d}.tmp", f"{GED}/c{a:06d}.npz")
        return ("OK", a)
    except Exception as e:
        return ("ERR", a, f"{type(e).__name__}: {e}")

if todo:
    with get_context("fork").Pool(8) as pool:
        done = 0
        for r in pool.imap_unordered(fe_work, todo, chunksize=1):
            if r[0] == "OK":
                done += 1
                print(f"  fe {done}/{len(todo)} @ {r[1]}", flush=True)
            else:
                raise SystemExit(f"FE ERR {r}")

X = np.empty((N, NF), dtype=np.float32)
for ck in TK:
    with np.load(f"{GED}/c{ck[0]:06d}.npz") as z:
        X[ck[0]:ck[1]] = z["X"]
print(f"[{(time.time()-t0)/60:.1f} min] FE assembled {X.shape}", flush=True)

# ---- persist v4 parquet (for future retraining without FE) ----
cols = {"s1_id": s1_arr, "cand_id": cand_arr, "y": y}
for j, fn in enumerate(V4):
    cols[fn] = X[:, j]
pl.DataFrame(cols).write_parquet("/home/ubuntu/train_feats_b100_v4.parquet")
print(f"[{(time.time()-t0)/60:.1f} min] wrote train_feats_b100_v4.parquet", flush=True)

# ---- split (exact notebook: sort unique, shuffle rng(0), 80/20) ----
u = np.sort(np.unique(s1_arr))
rng = np.random.default_rng(0)
rng.shuffle(u)
cut = int(0.8 * len(u))
va_list = u[cut:].tolist()
va_set = set(va_list)
is_va = np.isin(s1_arr, u[cut:])
Xtr, ytr = X[~is_va], y[~is_va]
Xva, yva = X[is_va], y[is_va]
s1_va, cand_va = s1_arr[is_va], cand_arr[is_va]
del X, s1_arr, cand_arr
gc.collect()
spw = float((ytr == 0).sum() / max(int((ytr == 1).sum()), 1))
print(f"train {Xtr.shape[0]:,} (pos {int(ytr.sum()):,}) | val {Xva.shape[0]:,} (pos {int(yva.sum()):,}) | spw {spw:.1f}", flush=True)

# ---- truth for val groups (FULL GT) ----
gt = pl.read_csv(f"{BASE}/train/train_ground_truth.tsv", separator="\t")
truth_ex = (gt.filter(pl.col(ic).is_in(va_list))
               .select([pl.col(ic).alias("s1_id"), pl.col(im).fill_null("").alias("m")])
               .select(["s1_id", pl.col("m").str.split(",").alias("m")])
               .explode("m")
               .select(["s1_id", pl.col("m").str.strip_chars().alias("cid")])
               .filter(pl.col("cid").str.len_chars() > 0))
del gt
truth = {}
for s, c in truth_ex.iter_rows():
    truth.setdefault(s, set()).add(c)
print(f"truth groups {len(truth):,} | pairs {truth_ex.height:,}", flush=True)

# val group row ranges (sort once)
order = np.argsort(s1_va, kind="stable")
s1_sorted = s1_va[order]
cand_sorted = cand_va[order]
starts = np.searchsorted(s1_sorted, va_list, side="left")
ends = np.searchsorted(s1_sorted, va_list, side="right")
del s1_va, cand_va, s1_sorted
gc.collect()

def macro_at(scores, thr):
    """scores aligned with cand_sorted rows."""
    Fs, mf = [], []
    tp = fp = fn = 0
    empty_n = emit_empty = 0
    for gi, s in enumerate(va_list):
        a, b = starts[gi], ends[gi]
        m = scores[a:b] >= thr
        p = set(cand_sorted[a:b][m].tolist())
        g = truth.get(s, set())
        inter = len(g & p)
        tp += inter; fp += len(p - g); fn += len(g - p)
        if not g:
            empty_n += 1
            emit_empty += 1 if p else 0
            Fs.append(0.0 if p else 1.0)
        else:
            P = inter / len(p) if p else 1.0
            R = inter / len(g)
            d = 0.25 * P + R
            F = 1.25 * P * R / d if d > 0 else 0.0
            Fs.append(F); mf.append(F)
    Pm = tp / max(tp + fp, 1); Rm = tp / max(tp + fn, 1)
    Fm = 1.25 * Pm * Rm / (0.25 * Pm + Rm) if (0.25 * Pm + Rm) > 0 else 0.0
    return float(np.mean(Fs)), float(np.mean(mf)), Fm, emit_empty

def best_tau(scores, lo=0.80, hi=0.996, step=0.005):
    best = (-1, None)
    for t in np.arange(lo, hi, step):
        m, _, _, _ = macro_at(scores, t)
        if m > best[0]: best = (m, float(t))
    return best

GRID_NOTE = np.arange(0.90, 0.996, 0.01)
def report(name, scores):
    scores = scores[order]
    m97, mm97, fm97, ee97 = macro_at(scores, 0.97)
    bm, bt = best_tau(scores)
    print(f"{name:16s} @0.97 macro={m97:.4f} matched={mm97:.4f} | best tau={bt:.3f} macro={bm:.4f} "
          f"| microF={fm97:.4f} empty-emit={ee97}", flush=True)
    return {"name": name, "m97": m97, "best_m": bm, "best_t": bt}

results = []

# ---- ARM A0: fresh XGBoost on the 20 baseline features (regression check vs 0.9082 anchor) ----
idx20 = [V4.index(f) for f in features.OLD]
m = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.03, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.9,
    reg_lambda=2.0, scale_pos_weight=spw,
    tree_method="hist", device="cpu", eval_metric="aucpr",
    random_state=42, n_jobs=8, early_stopping_rounds=50)
m.fit(Xtr[:, idx20], ytr, eval_set=[(Xva[:, idx20], yva)], verbose=False)
sc0 = m.predict_proba(Xva[:, idx20])[:, 1]
results.append(report("A0 xgb-20", sc0))
m.save_model("/home/ubuntu/v4_xgb20.json")
del m; gc.collect()

# ---- ARM A1: fresh xgb on 36 ----
m = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.03, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.9,
    reg_lambda=2.0, scale_pos_weight=spw,
    tree_method="hist", device="cpu", eval_metric="aucpr",
    random_state=42, n_jobs=8, early_stopping_rounds=50)
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
m.save_model("/home/ubuntu/v4_xgb.json")
sc1 = m.predict_proba(Xva)[:, 1]
results.append(report("A1 xgb-36", sc1))
xgb_sc = sc1
del m; gc.collect()

# ---- ARM A2: lgbm ----
ml = lgb.LGBMClassifier(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=2.0, scale_pos_weight=spw,
    random_state=42, n_jobs=8, verbose=-1, early_stopping_rounds=50)
ml.fit(Xtr, ytr, eval_set=[(Xva, yva)])
ml.booster_.save_model("/home/ubuntu/v4_lgbm.txt")
sc2 = ml.predict_proba(Xva)[:, 1]
results.append(report("A2 lgbm-36", sc2))
lgb_sc = sc2
del ml; gc.collect()

# ---- ARM A3: catboost (sqrt-balanced, per competitor) ----
mc = CatBoostClassifier(
    iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
    auto_class_weights="SqrtBalanced", random_seed=42,
    early_stopping_rounds=50, eval_metric="PRAUC",
    verbose=0, allow_writing_files=False, thread_count=8)
mc.fit(Xtr, ytr, eval_set=(Xva, yva))
mc.save_model("/home/ubuntu/v4_cat.cbm", format="cbm")
sc3 = mc.predict_proba(Xva)[:, 1]
results.append(report("A3 cat-36", sc3))
cat_sc = sc3
del mc, Xtr, ytr; gc.collect()

# ---- ARM A4/A5: ensembles ----
for name, w in (("A4 ens unif", [1/3, 1/3, 1/3]), ("A5 ens .4/.35/.25", [0.4, 0.35, 0.25])):
    ens = w[0]*xgb_sc + w[1]*lgb_sc + w[2]*cat_sc
    results.append(report(name, ens))

print("\n===== ranked (best tau macro) =====", flush=True)
for r in sorted(results, key=lambda r: -r["best_m"]):
    print(f"  {r['name']:16s} best={r['best_m']:.4f} @ t={r['best_t']:.3f} | @0.97={r['m97']:.4f}")
best = max(results, key=lambda r: r["best_m"])

# ---- write v4_model.json ----
if best["name"] == "A0 xgb-20":
    models = [{"kind": "xgb", "path": "/home/ubuntu/v4_xgb20.json"}]  # saved below if needed
    feats = features.OLD
elif best["name"] == "A1 xgb-36":
    models = [{"kind": "xgb", "path": "/home/ubuntu/v4_xgb.json", "weight": 1.0}]
    feats = V4
elif best["name"] == "A2 lgbm-36":
    models = [{"kind": "lgbm", "path": "/home/ubuntu/v4_lgbm.txt", "weight": 1.0}]
    feats = V4
elif best["name"] == "A3 cat-36":
    models = [{"kind": "cat", "path": "/home/ubuntu/v4_cat.cbm", "weight": 1.0}]
    feats = V4
elif best["name"] == "A4 ens unif":
    models = [{"kind": "xgb", "path": "/home/ubuntu/v4_xgb.json", "weight": 1/3},
              {"kind": "lgbm", "path": "/home/ubuntu/v4_lgbm.txt", "weight": 1/3},
              {"kind": "cat", "path": "/home/ubuntu/v4_cat.cbm", "weight": 1/3}]
    feats = V4
else:
    models = [{"kind": "xgb", "path": "/home/ubuntu/v4_xgb.json", "weight": 0.4},
              {"kind": "lgbm", "path": "/home/ubuntu/v4_lgbm.txt", "weight": 0.35},
              {"kind": "cat", "path": "/home/ubuntu/v4_cat.cbm", "weight": 0.25}]
    feats = V4

spec = {"feats": feats, "models": models, "threshold": best["best_t"],
        "arm": best["name"], "val_macro": best["best_m"],
        "anchor_0_9082": results[0]["m97"], "written": time.ctime()}
with open("/home/ubuntu/v4_model.json", "w") as fh:
    json.dump(spec, fh, indent=1)
print(f"\nWINNER: {best['name']} macro={best['best_m']:.4f} @ tau={best['best_t']}")
print(f"REGRESSION A0 @0.97: {results[0]['m97']:.4f} (baseline anchor 0.9082)")
print(f"wrote /home/ubuntu/v4_model.json | {(time.time()-t0)/60:.1f} min total")