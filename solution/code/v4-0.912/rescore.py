#!/usr/bin/env python3
# rescore.py - PREDICT-ONLY rescoring from saved X.npy. No FE, no blocking.
#   Recomputes pass.npz under a different threshold and/or model spec in minutes,
#   so threshold and post-processing sweeps cost only an assemble_submission run.
#
# env SRCD  = source dir with c*.cand.npz + c*.X.npy  (default /home/ubuntu/test_out_v4)
# env OUTD  = output dir: new c*.pass.npz + SYMLINKED c*.cand.npz (default /home/ubuntu/out_rescore)
# env MODEL = model spec json, same schema as v4_model.json
#             {"models":[{kind,path,weight}...], "feats":[...], "threshold":...}
# env TAU   = decision threshold (default = spec's own threshold)
# env NW    = fork workers (default 8)
#
# Then build TSVs:
#   SRCD is irrelevant for build; use:
#   OUTD=<this OUTD> OUTDIR=<...> POSTPROC=none|country|country+1to1 python assemble_submission.py
#
# Safety: OUTD must differ from SRCD (never clobbers the inference pass.npz).
# Regression check: run with TAU=<inference threshold> and MODEL=<inference model> -> pass
# totals must equal inference's.
import os, re, gc, glob, time, json
import numpy as np
from multiprocessing import get_context

SRCD = os.environ.get("SRCD", "/home/ubuntu/test_out_v4")
OUTD = os.environ.get("OUTD", "/home/ubuntu/out_rescore")
MODEL = os.environ.get("MODEL", "/home/ubuntu/v4_model.json")
NW = int(os.environ.get("NW", "8"))
if os.path.realpath(OUTD) == os.path.realpath(SRCD):
    raise SystemExit("ABORT: OUTD == SRCD would clobber inference pass.npz - pick another dir")
os.makedirs(OUTD, exist_ok=True)

# ---------- model spec (verbatim loader from inference.py) ----------
with open(MODEL) as fh:
    SPEC = json.load(fh)
TAU = float(os.environ.get("TAU", SPEC["threshold"]))

import features
COLIDX = np.array([features.FEATS.index(f) for f in SPEC["feats"]], dtype=np.int64)
KINDS = [m["kind"] for m in SPEC["models"]]
W = np.array([m.get("weight", 1.0) for m in SPEC["models"]], dtype=np.float64)
W /= W.sum()
import xgboost as xgb
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
print(f"rescore: model={MODEL} kinds={KINDS} w={[round(float(x), 3) for x in W]} "
      f"feats={len(COLIDX)} TAU={TAU} (spec thr={SPEC['threshold']}) | SRCD={SRCD} OUTD={OUTD}", flush=True)

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

# ---------- input inventory ----------
def chunk_id(path, suffix):
    return int(re.search(r"c(\d+)\." + suffix, os.path.basename(path)).group(1))

cand_files = sorted(glob.glob(f"{SRCD}/*.cand.npz"))
x_files = sorted(glob.glob(f"{SRCD}/*.X.npy"))
if len(cand_files) != 434 or len(x_files) != 434:
    raise SystemExit(f"ABORT: {len(cand_files)} cand / {len(x_files)} X (need 434 each - inference incomplete?)")
ids = [chunk_id(f, "cand") for f in cand_files]
if ids != [chunk_id(f, r"X\.npy") for f in x_files]:
    raise SystemExit("ABORT: cand/X chunk-id mismatch")

# symlink candidates into OUTD so assemble_submission sees 434 cand + 434 pass in one dir
nlink = 0
for f in cand_files:
    dst = os.path.join(OUTD, os.path.basename(f))
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(f), dst); nlink += 1
print(f"cand symlinks: {nlink} created | existing pass files to overwrite: "
      f"{len(glob.glob(OUTD + '/*.pass.npz'))}", flush=True)

# ---------- per-chunk work (fork-safe: numpy-only I/O, inference pattern) ----------
def work(job):
    n, src_cand, src_x = job
    tw = time.time()
    try:
        with np.load(src_cand) as z:
            ri_arr = z["ri"]; pid_arr = z["pid"]
        X = np.load(src_x)
        if X.shape[0] != ri_arr.size:
            return ("ERR", n, 0, 0, f"row misalignment X {X.shape[0]} != cand {ri_arr.size}")
        p = predict(X)
        keep = p >= TAU
        out_ri = ri_arr[keep]
        out_pid = pid_arr[keep]
        out_sc = p[keep]
        with open(f"{OUTD}/c{n:06d}.pass.tmp", "wb") as fh:
            np.savez(fh, ri=out_ri, pid=out_pid, score=out_sc)
        os.replace(f"{OUTD}/c{n:06d}.pass.tmp", f"{OUTD}/c{n:06d}.pass.npz")
        n_pass = out_ri.size
        del ri_arr, pid_arr, X, p, keep, out_ri, out_pid, out_sc
        gc.collect()
        return ("OK", n, n_pass, time.time() - tw, "")
    except Exception as e:
        return ("ERR", n, 0, 0, f"{type(e).__name__}: {e}")

jobs = [(i, f, x) for i, f, x in zip(ids, cand_files, x_files)]
t0 = time.time(); ok = errs = pass_tot = 0
with get_context("fork").Pool(NW) as pool:
    for r in pool.imap_unordered(work, jobs, chunksize=1):
        if r[0] == "OK":
            ok += 1; pass_tot += r[2]
        else:
            errs += 1
            print(f"ERR chunk c{r[1]:06d}: {r[4]}", flush=True)
        el = (time.time() - t0) / 60
        rate = (ok + errs) / el if el else 0
        eta = (434 - ok - errs) / rate if rate else 0
        if ok % 50 == 0 or r[0] == "ERR" or ok + errs == 434:
            print(f"[{ok + errs}/434] pass={pass_tot:,} err={errs} {el:.1f}m ETA {eta:.1f}m", flush=True)
print(f"rescore done: ok={ok} err={errs} pass={pass_tot:,} "
      f"tau={TAU} elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
if errs:
    raise SystemExit("ABORT: chunk errors - do NOT run assemble_submission from this OUTD")
print(f"next: OUTD={OUTD} OUTDIR=output POSTPROC=none python assemble_submission.py", flush=True)