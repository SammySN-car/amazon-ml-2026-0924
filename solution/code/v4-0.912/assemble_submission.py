#!/usr/bin/env python3
# assemble_submission.py - convert test_out npz checkpoints into the two submission TSVs.
# env OUTD   = dir with c*.npz (default /home/ubuntu/test_out; v4 uses test_out_v4)
# env OUTDIR = output dir     (default /home/ubuntu/output)
# env POSTPROC = none | country | country+1to1   (matching_results only; candidates
#   stay raw blocking output; matched stays a subset of candidates either way).
# Run AFTER inference.py prints "inference done".
import os, re, gc, glob, time
import numpy as np
import polars as pl

BASE = "/home/ubuntu/dataset/student_resource/dataset"
OUTD = os.environ.get("OUTD", "/home/ubuntu/test_out")
OUT = os.environ.get("OUTDIR", "/home/ubuntu/output")
POSTPROC = os.environ.get("POSTPROC", "none")
os.makedirs(OUT, exist_ok=True)

def pick(cols, *pats):
    for p in pats:
        for c in cols:
            if p.lower() in c.lower():
                return c
    raise ValueError(f"{pats} not in {cols}")

t1 = pl.read_csv(f"{BASE}/test/test_source1.tsv", separator="\t")
idc1 = pick(t1.columns, "entity_id", "id")
ctc1 = pick(t1.columns, "country")
s1_ids = t1[idc1].to_list()
ct_s1 = t1[ctc1].fill_null("").to_numpy()
del t1
t2 = pl.read_csv(f"{BASE}/test/test_source2.tsv", separator="\t")
idc2 = pick(t2.columns, "entity_id", "id")
ctc2 = pick(t2.columns, "country")
e2 = t2[idc2].to_numpy(); ct2 = t2[ctc2].fill_null("").to_numpy()
del t2
t3 = pl.read_csv(f"{BASE}/test/test_source3.tsv", separator="\t")
idc3 = pick(t3.columns, "entity_id", "id")
ctc3 = pick(t3.columns, "country")
e3 = t3[idc3].to_numpy(); ct3 = t3[ctc3].fill_null("").to_numpy()
del t3
NT2 = len(e2)
ents = np.concatenate([e2, e3])
ct_e = np.concatenate([ct2, ct3])
n = len(s1_ids)
print(f"OUTD={OUTD} | POSTPROC={POSTPROC} | s1 {n:,} | ents {len(ents):,}", flush=True)
del e2, e3, ct2, ct3
gc.collect()

def collect(suffix, want_score=False):
    files = glob.glob(f"{OUTD}/*.{suffix}.npz")
    if len(files) != 434:
        raise SystemExit(f"ABORT: {len(files)} {suffix} files (need 434 - inference incomplete?)")
    files.sort(key=lambda p: int(re.search(r"c(\d+)\.", os.path.basename(p)).group(1)))
    ris, pids, scs = [], [], []
    for fpath in files:
        with np.load(fpath) as z:
            ris.append(z["ri"]); pids.append(z["pid"])
            if want_score and "score" in z.files:
                scs.append(z["score"])
    ri = np.concatenate(ris); pid = np.concatenate(pids)
    del ris, pids
    if ri.size:
        if np.any(np.diff(ri) < 0) or ri.min() < 0 or ri.max() >= n:
            raise SystemExit(f"ABORT: ri not monotone/in-range in {suffix}")
        if pid.min() < 0 or pid.max() >= len(ents):
            raise SystemExit(f"ABORT: pid out of range in {suffix}")
    print(f"  {suffix}: {ri.size:,} pairs", flush=True)
    sc = np.concatenate(scs) if scs and len(scs) == len(files) else None
    return ri, pid, sc

def apply_postproc(ri, pid, sc):
    if POSTPROC == "none":
        return ri, pid
    mask = ct_s1[ri] == ct_e[pid]
    cut = int((~mask).sum())
    ri, pid = ri[mask], pid[mask]
    if sc is not None:
        sc = sc[mask]
    print(f"  country filter: dropped {cut:,} cross-country matches", flush=True)
    if POSTPROC == "country+1to1":
        if sc is None:
            raise SystemExit("ABORT: country+1to1 needs scores in pass.npz")
        order = np.argsort(-sc, kind="stable")
        _, firsts = np.unique(pid[order], return_index=True)
        keep = np.zeros(pid.size, dtype=bool)
        keep[order[firsts]] = True
        print(f"  1to1 resolve: dropped {int((~keep).sum()):,} conflicting matches", flush=True)
        ri, pid = ri[keep], pid[keep]
    return ri, pid

def write_tsv(path, header, ri, pid):
    t0 = time.time()
    counts = np.bincount(ri, minlength=n) if ri.size else np.zeros(n, dtype=np.int64)
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    strs = ents[pid] if pid.size else np.empty(0, dtype=ents.dtype)
    nonempty = int((counts > 0).sum())
    with open(path, "w", encoding="utf-8", buffering=1 << 24) as fh:
        fh.write(header)
        for i in range(n):
            c = int(counts[i])
            if c == 0:
                fh.write(s1_ids[i] + "\t\n")
            else:
                fh.write(s1_ids[i] + "\t" + ",".join(strs[starts[i]:starts[i + 1]].tolist()) + "\n")
    print(f"  wrote {path} | rows {n:,} | non-empty {nonempty:,} | ids {int(counts.sum()):,} | {time.time() - t0:.0f}s", flush=True)

print("pass -> matching_results.tsv ...", flush=True)
ri, pid, sc = collect("pass", want_score=True)
ri, pid = apply_postproc(ri, pid, sc)
del sc
write_tsv(f"{OUT}/matching_results.tsv", "source1_entity_id\tmatched_entity_ids\n", ri, pid)
del ri, pid
gc.collect()

print("cand -> candidate_pairs.tsv ...", flush=True)
ri, pid, _ = collect("cand")
write_tsv(f"{OUT}/candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\n", ri, pid)
print("assemble_submission done", flush=True)