#!/usr/bin/env python3
"""prepare_data.py - builds TRAIN blocking candidates from raw train data (no FE, no model).
Chain: features.py (lib) -> prepare_data.py (this) -> train.py -> inference.py -> assemble_submission.py
Sample : seed-0 random 64,125 S1 rows (same size as the original training sample;
the original sample was EDA-derived - see README, exact replication not required).
Blocking: same key family as inference.py (N-norm tokens, NFD accent-fold exact key,
postings restricted to sample token universe, MAXB=200, greedy budget=100).
One deliberate difference: inference.py additionally partitions candidates
by country at test time; the train pool does not (the shipped model was
trained on this asymmetry - see README "Blocking").
Output : train_cands.parquet (s1_id, cand_id) - consumed by train.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")   # must be set BEFORE polars import
import gc, time, unicodedata
import numpy as np, polars as pl
from multiprocessing import get_context

BASE = "/home/ubuntu/dataset/student_resource/dataset"
OUT = os.environ.get("CANDS_OUT", "/home/ubuntu/train_cands.parquet")
N_SAMPLE, SEED = 64_125, 0
NW, BUDGET, CHUNK = 8, 100, 4000

# ---------- helpers (exact copies from inference.py) ----------
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

# ---------- load + seeded sample ----------
t0 = time.time()
t1 = pl.read_csv(f"{BASE}/train/train_source1.tsv", separator="\t")
assert N_SAMPLE <= t1.height, (N_SAMPLE, t1.height)
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(t1.height, N_SAMPLE, replace=False))
samp = t1.with_row_index("ri").filter(pl.col("ri").is_in(idx.tolist())).drop("ri")
del t1; gc.collect()
print(f"sample {samp.height:,} / seed {SEED} | {((time.time()-t0)/60):.1f} min", flush=True)

idc = pick(samp.columns, "entity_id", "id")
nc, ac, cc = pick(samp.columns, "business_name", "name"), pick(samp.columns, "business_address", "address"), pick(samp.columns, "country")

s1f = (samp.select([pl.col(idc).alias("s1_id"), pl.col(cc).alias("ct"), pl.col(nc).alias("nA"), pl.col(ac).alias("aA")])
        .with_columns(N(pl.col("nA")).alias("sn1"))
        .with_columns([pl.col("sn1").str.extract_all(r"\w+").list.unique().alias("n1"),
                       N(pl.col("aA")).alias("na")])
        .with_columns([pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk"),
                       pl.col("na").str.extract_all(r"\w+").list.unique().alias("a1"),
                       pl.col("na").str.extract_all(r"\d{2,}").list.unique().alias("m1")])
        .drop("na"))
del samp; gc.collect()
print(f"[{((time.time()-t0)/60):.1f} min] s1f {s1f.height:,}", flush=True)

# ---------- token universe = sampled S1 rows (mirrors eda.py) ----------
tokN_l = s1f.select(pl.col("n1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokA_l = s1f.select(pl.col("a1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokM_l = s1f.select(pl.col("m1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
fk_list = s1f.select("fk").unique()["fk"].to_list()
print(f"[{((time.time()-t0)/60):.1f} min] universe {len(tokN_l)}/{len(tokA_l)}/{len(tokM_l)} | exact {len(fk_list)}", flush=True)

# ---------- pool = train source2 + source3 ----------
t2 = pl.read_csv(f"{BASE}/train/train_source2.tsv", separator="\t")
t3 = pl.read_csv(f"{BASE}/train/train_source3.tsv", separator="\t")
nc2, ac2 = pick(t2.columns, "business_name", "name"), pick(t2.columns, "business_address", "address")
nc3, ac3 = pick(t3.columns, "business_name", "name"), pick(t3.columns, "business_address", "address")
i2, i3 = pick(t2.columns, "entity_id", "id"), pick(t3.columns, "entity_id", "id")
NT2 = t2.height
ids_p = np.array(t2[i2].fill_null("").to_list() + t3[i3].fill_null("").to_list(), dtype=object)
print(f"[{((time.time()-t0)/60):.1f} min] pool ids {len(ids_p):,}", flush=True)

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

# ---------- blocking functions (verbatim from inference.py) ----------
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
    n_t = [t for t in (row["n1"] or []) if t in PT_N]
    a_t = [t for t in (row["a1"] or []) if t in PT_A]
    m_t = [t for t in (row["m1"] or []) if t in PT_M]
    ev = []
    for kind, toks in (("N", n_t), ("A", a_t), ("M", m_t)):
        for t in toks:
            ev.append((len(PMAP[kind][t]), PMAP[kind][t], True))
    ex = EXT.get(row["fk"])
    if ex is not None and len(ex) > 0:
        ev.append((len(ex), ex, True))
    for arr, kind in ((n_t, "N"), (a_t, "A"), (m_t, "M")):
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                r = eval_pair(kind, arr[i], kind, arr[j])
                if r: ev.append((r[0], r[1], False))
    for x, kx in ((n_t, "N"), (a_t, "A")):
        for y, ky in ((a_t, "A"), (m_t, "M")):
            if kx == ky: continue
            for t1_ in x:
                for t2_ in y:
                    r = eval_pair(kx, t1_, ky, t2_)
                    if r: ev.append((r[0], r[1], False))
    return ev

def run_greedy(ev, budget):
    # ev: (freq, arr, is-solo-key)
    c = [(df, arr) for df, arr, _solo in ev]
    c.sort(key=lambda x: x[0])
    u = set()
    for df, arr in c:
        if df > budget: break
        add = arr[~np.isin(arr, list(u))] if u else arr
        if len(u) + add.size <= budget:
            u.update(add.tolist())
    return u

# ---------- chunks (blocking only - no FE, no model) ----------
n_rows = s1f.height
CH = [(a, min(a + CHUNK, n_rows)) for a in range(0, n_rows, CHUNK)]
print(f"[{((time.time()-t0)/60):.1f} min] chunks {len(CH)} | NW={NW} | budget={BUDGET}", flush=True)

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
        del ri_l, pid_l
        return ("OK", a, ri_arr, pid_arr, time.time() - tw)
    except Exception as e:
        return ("ERR", a, None, None, f"{type(e).__name__}: {e}")

t1e = time.time(); ok = errs = 0; tot = 0; RIS, PIDS, err_starts = [], [], []
with get_context("fork").Pool(NW) as pool:
    for r in pool.imap_unordered(work, CH, chunksize=1):
        if r[0] == "OK":
            ok += 1; tot += r[2].size
            if r[2].size:
                RIS.append(r[2]); PIDS.append(r[3])
            el = time.time() - t1e
            rem = (len(CH) - ok) * el / ok
            print(f"[{ok}/{len(CH)}] chunk@{r[1]} cands={r[2].size:,} {r[4]:.0f}s | tot={tot:,} | {el/60:.1f}m | ETA {rem/60:.1f}m", flush=True)
        else:
            errs += 1; err_starts.append(r[1])
            print(f"ERR chunk {r[1]}: {r[4]}", flush=True)

if errs:
    raise SystemExit(f"FAILED chunks {err_starts} - blocking incomplete, re-run to retry")

ri = np.concatenate(RIS) if RIS else np.zeros(0, np.int64)
pid = np.concatenate(PIDS) if PIDS else np.zeros(0, np.int32)
s1_ids = s1f["s1_id"].to_numpy()
df = pl.DataFrame({"s1_id": s1_ids[ri], "cand_id": ids_p[pid]})
with open(OUT + ".tmp", "wb") as fh:
    df.write_parquet(fh)
os.replace(OUT + ".tmp", OUT)
mean_c = df.height / max(n_rows, 1)
print(f"\nprepare_data done: {df.height:,} pairs | {n_rows:,} S1s | mean {mean_c:.1f}/S1 "
      f"(original b100: 5,707,697 / 64,125 = 89.0) | {(time.time()-t1e)/60:.1f} min blocking", flush=True)
print(f"wrote {OUT} | {(time.time()-t0)/60:.1f} min total", flush=True)