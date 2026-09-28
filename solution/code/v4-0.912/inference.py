#!/usr/bin/env python3
# inference.py - test pipeline v4: country-partitioned greedy, 36 features (features),
# ensemble scoring from v4_model.json, per-chunk X.npy save (future rescoring = predict-only).
# env BASE = dataset root (contains train/ and test/); env OUTD = checkpoint dir
# (default /home/ubuntu/test_out_v4); env MODEL = model spec path. The baseline
# pipeline keeps its own OUTD dir. Resumable per chunk.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
import re, gc, json, time, glob, sys
import numpy as np, polars as pl
import xgboost as xgb
from multiprocessing import get_context
import features

BASE = os.environ.get("BASE", "/home/ubuntu/dataset/student_resource/dataset")
OUTD = os.environ.get("OUTD", "/home/ubuntu/test_out_v4")
NW, BUDGET, CHUNK = 8, 100, 4000
os.makedirs(OUTD, exist_ok=True)
for f in glob.glob(OUTD + "/*.tmp"):
    os.remove(f)

# ---------- helpers (state build; copied from baseline-0.924/inference.py) ----------
def N(e):
    return (e.str.to_uppercase()
             .str.replace_all(r"[!-/:-@\[-`{-~]", "")
             .str.replace_all(r"\s+", " ")
             .str.strip_chars())

def fold(s):
    if s is None: return None
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))

def pick(cols, *pats):
    for p in pats:
        for c in cols:
            if p.lower() in c.lower(): return c
    raise ValueError(f"{pats} not in {cols}")

import unicodedata

# ---------- model spec ----------
with open(os.environ.get("MODEL", "/home/ubuntu/v4_model.json")) as fh:
    SPEC = json.load(fh)
COLIDX = np.array([features.FEATS.index(f) for f in SPEC["feats"]], dtype=np.int64)
THR = SPEC["threshold"]
KINDS = [m["kind"] for m in SPEC["models"]]
W = np.array([m.get("weight", 1.0) for m in SPEC["models"]], dtype=np.float64)
W /= W.sum()
import lightgbm as lgb
from catboost import CatBoostClassifier
MODELS = []
for m in SPEC["models"]:
    if m["kind"] == "xgb":
        mdl = xgb.XGBClassifier(); mdl.load_model(m["path"])
    elif m["kind"] == "lgbm":
        mdl = lgb.Booster(model_file=m["path"])
    elif m["kind"] == "cat":
        mdl = CatBoostClassifier(); mdl.load_model(m["path"], format="cbm"); mdl.set_thread_count(1)
    else:
        raise ValueError(m["kind"])
    MODELS.append(mdl)
print(f"model spec: {len(MODELS)} models {KINDS} w={[round(float(x),3) for x in W]} "
      f"thr={THR} feats={len(COLIDX)}", flush=True)

def predict(X):
    Xs = X[:, COLIDX]
    acc = None
    for i, mdl in enumerate(MODELS):
        if KINDS[i] == "xgb":
            p = mdl.predict_proba(Xs)[:, 1]
        elif KINDS[i] == "lgbm":
            p = np.asarray(mdl.predict(Xs)).reshape(-1)
        else:
            # CatBoost: predict() returns hard class labels; probabilities match train.py's gate
            p = mdl.predict_proba(Xs)[:, 1]
        acc = W[i] * p if acc is None else acc + W[i] * p
    return acc.astype(np.float32)

# ---------- state build (copied from baseline-0.924/inference.py) ----------
t0 = time.time()
t1 = pl.read_csv(f"{BASE}/test/test_source1.tsv", separator="\t")
t2 = pl.read_csv(f"{BASE}/test/test_source2.tsv", separator="\t")
t3 = pl.read_csv(f"{BASE}/test/test_source3.tsv", separator="\t")
assert (t1.height, t2.height, t3.height) == (1732544, 4887273, 5082316), (t1.height, t2.height, t3.height)
idc  = pick(t1.columns, "entity_id", "id")
nc, ac, cc = pick(t1.columns, "business_name", "name"), pick(t1.columns, "business_address", "address"), pick(t1.columns, "country")
nc2, ac2 = pick(t2.columns, "business_name", "name"), pick(t2.columns, "business_address", "address")
nc3, ac3 = pick(t3.columns, "business_name", "name"), pick(t3.columns, "business_address", "address")
c2c, c3c = pick(t2.columns, "country"), pick(t3.columns, "country")

s1f = (t1.select([pl.col(idc).alias("s1_id"), pl.col(cc).alias("ct"), pl.col(nc).alias("nA"), pl.col(ac).alias("aA")])
        .with_columns(N(pl.col("nA")).alias("sn1"))
        .with_columns([pl.col("sn1").str.extract_all(r"\w+").list.unique().alias("n1"),
                       N(pl.col("aA")).alias("na")])
        .with_columns([pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk"),
                       pl.col("na").str.extract_all(r"\w+").list.unique().alias("a1"),
                       pl.col("na").str.extract_all(r"\d{2,}").list.unique().alias("m1")])
        .drop("na"))
del t1; gc.collect()
print(f"[{((time.time()-t0)/60):.1f} min] s1f {s1f.height:,}", flush=True)

tokN_l = s1f.select(pl.col("n1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokA_l = s1f.select(pl.col("a1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokM_l = s1f.select(pl.col("m1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
fk_list = s1f.select("fk").unique()["fk"].to_list()
print(f"[{((time.time()-t0)/60):.1f} min] universe {len(tokN_l)}/{len(tokA_l)}/{len(tokM_l)} | exact {len(fk_list)}", flush=True)

NT2 = t2.height
names_p = np.array(t2[nc2].fill_null("").to_list() + t3[nc3].fill_null("").to_list(), dtype=object)
addrs_p = np.array(t2[ac2].fill_null("").to_list() + t3[ac3].fill_null("").to_list(), dtype=object)
cts_p   = np.array(t2[c2c].fill_null("").to_list() + t3[c3c].fill_null("").to_list(), dtype=object)
# int country codes: fast cf (object-array string compare was 20min+/chunk)
_ct_uniq, _cts_i = np.unique(cts_p, return_inverse=True)
CTI = _cts_i.astype(np.int32)
CT_CODE = {s: i for i, s in enumerate(_ct_uniq.tolist())}
del _cts_i
print(f"[{((time.time()-t0)/60):.1f} min] pool arrays {len(names_p):,}", flush=True)

PT_N, PT_A, PT_M, EXT = {}, {}, {}, {}
def merge(dst, groups):
    for key, arr in groups:
        dst[key] = np.concatenate([dst[key], arr]) if key in dst else arr

for off, (src, ncx, acx) in enumerate([(t2, nc2, ac2), (t3, nc3, ac3)]):
    base = (src.with_row_index("pid")
              .with_columns(N(pl.col(ncx)).alias("nm"))
              .with_columns(pl.col("nm").map_elements(fold, return_dtype=pl.Utf8).alias("fk"))
              .select(["pid", "nm", "fk"]))
    ex = base.filter(pl.col("fk").is_in(fk_list)).select(["fk", pl.col("pid").alias("rid")])
    if off: ex = ex.with_columns((pl.col("rid") + NT2).alias("rid"))
    merge(EXT, ((k, np.asarray(r, dtype=np.int32)) for k, r in ex.group_by("fk").agg(pl.col("rid").sort()).iter_rows()))
    del ex
    pn = (base.select(["pid", pl.col("nm").str.extract_all(r"\w+").alias("t")])
              .explode("t").filter(pl.col("t").is_in(tokN_l))
              .select(["t", pl.col("pid").alias("rid")]))
    if off: pn = pn.with_columns((pl.col("rid") + NT2).alias("rid"))
    merge(PT_N, ((k, np.asarray(r, dtype=np.int32)) for k, r in pn.group_by("t").agg(pl.col("rid").sort()).iter_rows()))
    del base, pn; gc.collect()
    ba = (src.with_row_index("pid").with_columns(N(pl.col(acx)).alias("na")).select(["pid", "na"]))
    for dst, pat, toks in ((PT_A, r"\w+", tokA_l), (PT_M, r"\d{2,}", tokM_l)):
        pa = (ba.select(["pid", pl.col("na").str.extract_all(pat).alias("t")])
                .explode("t").filter(pl.col("t").is_in(toks))
                .select(["t", pl.col("pid").alias("rid")]))
        if off: pa = pa.with_columns((pl.col("rid") + NT2).alias("rid"))
        merge(dst, ((k, np.asarray(r, dtype=np.int32)) for k, r in pa.group_by("t").agg(pl.col("rid").sort()).iter_rows()))
        del pa
    del ba; gc.collect()
    print(f"[{((time.time()-t0)/60):.1f} min] pool source {off} | N {len(PT_N)} A {len(PT_A)} M {len(PT_M)} EX {len(EXT)}", flush=True)

del t2, t3; gc.collect()
print(f"[{((time.time()-t0)/60):.1f} min] posting rows {sum(len(v) for v in PT_N.values()):,} "
      f"{sum(len(v) for v in PT_A.values()):,} {sum(len(v) for v in PT_M.values()):,}", flush=True)

# ---------- blocking (country-partitioned greedy, same keys as baseline-0.924) ----------
MAXB = 200
PMAP = {"N": PT_N, "A": PT_A, "M": PT_M}

def get_b(kind, t):
    a = PMAP[kind].get(t)
    return None if a is None or len(a) == 0 else a

def inter_count(a, b):
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a)
    ok = idx < len(b)
    return int(np.count_nonzero(ok & (b[np.minimum(idx, len(b) - 1)] == a)))

def eval_pair(k1, t1_, k2, t2_):
    a, b = get_b(k1, t1_), get_b(k2, t2_)
    if a is None or b is None: return None
    df = inter_count(a, b)
    if df == 0 or df > MAXB: return None
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a); ok = idx < len(b)
    hits = a[ok & (b[np.minimum(idx, len(b) - 1)] == a)]
    return (df, hits) if hits.size else None

def build_keys_t(row):
    # v4: candidates are country-partitioned (train GT: 0 cross-country pairs)
    ct = row["ct"]
    ctc = CT_CODE.get(ct, -1)
    def cf(arr):
        return arr[CTI[arr] == ctc] if ctc >= 0 else arr[:0]
    n_t = [t for t in (row["n1"] or []) if t in PT_N]
    a_t = [t for t in (row["a1"] or []) if t in PT_A]
    m_t = [t for t in (row["m1"] or []) if t in PT_M]
    ev = []
    for kind, toks in (("N", n_t), ("A", a_t), ("M", m_t)):
        for t in toks:
            h = cf(PMAP[kind][t])
            if h.size: ev.append((len(PMAP[kind][t]), h, True))
    ex = EXT.get(row["fk"])
    if ex is not None and len(ex) > 0:
        h = cf(ex)
        if h.size: ev.append((len(ex), h, True))
    for arr, kind in ((n_t, "N"), (a_t, "A"), (m_t, "M")):
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                r = eval_pair(kind, arr[i], kind, arr[j])
                if r:
                    h = cf(r[1])
                    if h.size: ev.append((r[0], h, False))
    for x, kx in ((n_t, "N"), (a_t, "A")):
        for y, ky in ((a_t, "A"), (m_t, "M")):
            if kx == ky: continue
            for t1_ in x:
                for t2_ in y:
                    r = eval_pair(kx, t1_, ky, t2_)
                    if r:
                        h = cf(r[1])
                        if h.size: ev.append((r[0], h, False))
    return ev

def run_greedy(ev, budget, solo_only=False):
    c = [(df, arr) for df, arr, solo in ev if (solo or not solo_only)]
    c.sort(key=lambda x: x[0])
    u = set()
    for df, arr in c:
        if df > budget: break
        add = arr[~np.isin(arr, list(u))] if u else arr
        if len(u) + add.size <= budget:
            u.update(add.tolist())
    return u

NARR = s1f["nA"].to_numpy()
AARR = s1f["aA"].to_numpy()
CTARR = s1f["ct"].to_numpy()

# ---------- chunks ----------
n_rows = s1f.height
CH = [(a, min(a + CHUNK, n_rows)) for a in range(0, n_rows, CHUNK)]
todo = [c for c in CH if not (os.path.exists(f"{OUTD}/c{c[0]:06d}.pass.npz")
                              and os.path.exists(f"{OUTD}/c{c[0]:06d}.X.npy"))]
print(f"[{((time.time()-t0)/60):.1f} min] chunks {len(CH)} | todo {len(todo)} | thr={THR} | NW={NW}", flush=True)
if not todo:
    print("nothing to do - all chunks complete", flush=True)
    raise SystemExit(0)

def work(ck):
    a, b = ck
    tw = time.time()
    try:
        sl = s1f.slice(a, b - a)
        ri_l, pid_l = [], []
        for ri_off, row in enumerate(sl.iter_rows(named=True)):
            ev = build_keys_t(row)
            u = run_greedy(ev, BUDGET) if ev else set()
            if u:
                g = a + ri_off
                ri_l.extend([g] * len(u)); pid_l.extend(u)
        ri_arr = np.asarray(ri_l, dtype=np.int64)
        pid_arr = np.asarray(pid_l, dtype=np.int32)
        n_cand = ri_arr.size
        del ri_l, pid_l
        with open(f"{OUTD}/c{a:06d}.cand.tmp", "wb") as fh:
            np.savez(fh, ri=ri_arr, pid=pid_arr)
        os.replace(f"{OUTD}/c{a:06d}.cand.tmp", f"{OUTD}/c{a:06d}.cand.npz")

        X = np.zeros((n_cand, features.NF), dtype=np.float32)
        for i in range(n_cand):
            ri = int(ri_arr[i]); pid = int(pid_arr[i])
            X[i] = features.featurize(NARR[ri], names_p[pid], AARR[ri], addrs_p[pid],
                                     CTARR[ri], cts_p[pid])
        # save X: future model/threshold changes = predict-only, no FE rerun
        with open(f"{OUTD}/c{a:06d}.X.tmp", "wb") as fh:
            np.save(fh, X)
        os.replace(f"{OUTD}/c{a:06d}.X.tmp", f"{OUTD}/c{a:06d}.X.npy")

        if n_cand:
            sc = predict(X)
            keep = sc >= THR
            out_ri = ri_arr[keep]
            out_pid = pid_arr[keep]
            out_sc = sc[keep].astype(np.float32)
        else:
            out_ri = np.zeros(0, np.int64)
            out_pid = np.zeros(0, np.int32)
            out_sc = np.zeros(0, np.float32)
        with open(f"{OUTD}/c{a:06d}.pass.tmp", "wb") as fh:
            np.savez(fh, ri=out_ri, pid=out_pid, score=out_sc)
        os.replace(f"{OUTD}/c{a:06d}.pass.tmp", f"{OUTD}/c{a:06d}.pass.npz")
        n_pass = out_ri.size
        del ri_arr, pid_arr, out_ri, out_pid, out_sc, X
        gc.collect()
        return ("OK", a, n_cand, n_pass, time.time() - tw)
    except Exception as e:
        return ("ERR", a, 0, 0, f"{type(e).__name__}: {e}")

# ---------- run ----------
t1e = time.time(); ok = errs = 0; cand_tot = pass_tot = 0; err_starts = []
with get_context("fork").Pool(NW) as pool:
    for r in pool.imap_unordered(work, todo, chunksize=1):
        if r[0] == "OK":
            ok += 1; cand_tot += r[2]; pass_tot += r[3]
            el = time.time() - t1e
            rem = (len(todo) - ok) * el / ok
            print(f"[{ok}/{len(todo)}] chunk@{r[1]} cands={r[2]:,} pass={r[3]:,} {r[4]:.0f}s | "
                  f"tot cands={cand_tot:,} pass={pass_tot:,} | {el/60:.1f}m | ETA {rem/3600:.2f}h", flush=True)
        else:
            errs += 1; err_starts.append(r[1])
            print(f"ERR chunk {r[1]}: {r[2] if isinstance(r[2], str) else r[4]}", flush=True)

print(f"\ninference done: ok={ok} err={errs} | cands {cand_tot:,} pass {pass_tot:,} | {(time.time()-t1e)/3600:.1f} h", flush=True)
if errs:
    print("failed chunks (re-run to retry):", err_starts, flush=True)