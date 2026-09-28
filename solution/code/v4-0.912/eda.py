"""
eda.py - EDA phase only, Amazon ML Challenge 2026 Business Entity Resolution.
Extracted from training.ipynb: code cells 0-22 (setup, EDA-01..EDA-16,
blocking-state save/restore, PMAP guard). BUILD/training cells excluded.
Notebook magics (!pip, %%time) are commented out so this parses as plain Python.
Edit the data paths at the top for your machine.
Companion doc: eda_walkthrough.md (what each EDA asked and how it was found).
"""

# --- cell 0 ---
# [notebook shell] !pip install -q polars numpy pandas huggingface_hub rapidfuzz

# verify (optional)
import polars, numpy, pandas, rapidfuzz
print(f"polars {polars.__version__} | numpy {numpy.__version__} | pandas {pandas.__version__} | rapidfuzz {rapidfuzz.__version__}")

# --- cell 1 ---
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="theRavish/amazon-ml",
    repo_type="dataset",
    local_dir="/home/ubuntu/dataset",
    token=os.environ["HF_TOKEN"]  # export HF_TOKEN before running; never hardcode tokens
)

# --- cell 2 ---
import pandas as pd

s1 = pd.read_csv("/home/ubuntu/dataset/student_resource/dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("/home/ubuntu/dataset/student_resource/dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("/home/ubuntu/dataset/student_resource/dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("/home/ubuntu/dataset/student_resource/dataset/train/train_ground_truth.tsv", sep="\t")

print(s1.shape)
print(s2.shape)
print(s3.shape)
print(gt.shape)

# --- cell 3 ---
print(s1.shape)
print(s2.shape)
print(s3.shape)
print(gt.shape)

print(s1.columns)
print(s1.head())

print(s2.columns)
print(s2.head())

print(s3.columns)
print(s3.head())

print(gt.columns)
print(gt.head())

print(s1.dtypes)
print(gt.dtypes)

# --- cell 4 ---
# [notebook shell] !pip install -q rapidfuzz
import numpy as np, pandas as pd, re, gc
from rapidfuzz import fuzz

# 0. confirm actual schema first
print("gt columns:", list(gt.columns), "| gt shape:", gt.shape)
print("s1 columns:", list(s1.columns))
print(gt.head(3).to_string())

rng = np.random.default_rng(0)
N_POS, N_NEG = 25000, 40000          # SAMPLE-BASED results (labeled as such)

def norm(s):  return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", str(s))).upper().strip()

# 1. derive n_matches LOCALLY (gt untouched), sample rows, explode ONLY the sample
m_full = gt["matched_entity_ids"].fillna("")
n_matches = (m_full.str.count(",") + 1).where(m_full != "", 0)   # local Series
print("sanity: zero-match =", int((n_matches==0).sum()),
      "| total pairs =", int(n_matches.sum()), "| max =", int(n_matches.max()))

p = n_matches.to_numpy(dtype=float); p /= p.sum()
samp = gt.iloc[rng.choice(len(gt), N_POS, p=p)]
ex = samp.assign(m=samp["matched_entity_ids"].fillna("").str.split(",")).explode("m")
ex = ex[ex["m"].notna() & (ex["m"] != "")]
pairs_pos = ex[["source1_entity_id", "m"]].rename(columns={"source1_entity_id": "s1", "m": "cand"})
print(f"sampled {N_POS} GT rows -> {len(pairs_pos)} positive pairs")

# 2. random same-country negatives (by construction)
pools = {}
for nm, df in [("S2", s2), ("S3", s3)]:
    for c in df["country"].unique():
        pools[(nm, str(c))] = df.loc[df["country"] == c, "entity_id"].to_numpy()
i = rng.choice(len(s1), N_NEG, replace=False)
s1samp = s1.iloc[i]
neg_rows = []
for eid, c in zip(s1samp["entity_id"], s1samp["country"]):
    nm = "S2" if rng.random() < 0.5 else "S3"
    arr = pools[(nm, str(c))]
    neg_rows.append((eid, arr[rng.integers(len(arr))]))
pairs_neg = pd.DataFrame(neg_rows, columns=["s1", "cand"])
del samp, ex, s1samp, neg_rows, p, m_full; gc.collect()

# 3. pull text ONLY for sampled ids
def assemble(prs):
    def pull(ids, df):
        return df[df["entity_id"].isin(ids)][["entity_id","business_name","business_address","country"]]
    s1t = pull(set(prs["s1"]), s1)
    m2 = prs.loc[prs["cand"].str[:2] == "S2", "cand"]; m3 = prs.loc[prs["cand"].str[:2] == "S3", "cand"]
    ct = pd.concat([pull(set(m2), s2), pull(set(m3), s3)])
    out = prs.merge(s1t, left_on="s1", right_on="entity_id") \
             .merge(ct, left_on="cand", right_on="entity_id", suffixes=("_1","_c"))
    return out.drop(columns=[c for c in out.columns if c.startswith("entity_id")])

pos = assemble(pairs_pos); neg = assemble(pairs_neg)
del pairs_pos, pairs_neg, pools; gc.collect()
print(f"pos text rows: {len(pos)} | neg text rows: {len(neg)} | country mismatch in pos: {(pos['country_1']!=pos['country_c']).mean():.6f}")

# 4. similarity features
def feats(df, label):
    n1 = [norm(x) for x in df["business_name_1"]]; n2 = [norm(x) for x in df["business_name_c"]]
    t1 = [frozenset(x.split()) for x in n1];        t2 = [frozenset(x.split()) for x in n2]
    a1 = [frozenset(norm(x).split()) if isinstance(x, str) else frozenset() for x in df["business_address_1"]]
    a2 = [frozenset(norm(x).split()) if isinstance(x, str) else frozenset() for x in df["business_address_c"]]
    r = pd.DataFrame({
        "label": label,
        "name_exact": np.fromiter((a==b for a,b in zip(n1,n2)), bool, len(df)),
        "name_ratio": np.fromiter((fuzz.ratio(a,b) for a,b in zip(n1,n2)), np.float64, len(df)),
        "name_jacc":  np.fromiter((len(a&b)/len(a|b) if a|b else 0.0 for a,b in zip(t1,t2)), np.float64, len(df)),
        "addr_miss":  np.fromiter((len(x)==0 or len(y)==0 for x,y in zip(a1,a2)), bool, len(df)),
        "addr_jacc":  np.fromiter((len(a&b)/len(a|b) if (a and b and a|b) else np.nan for a,b in zip(a1,a2)), np.float64, len(df)),
    })
    r["addr_exact"] = np.fromiter(
        ((not pd.isna(x)) and (not pd.isna(y)) and norm(x)==norm(y)
         for x,y in zip(df["business_address_1"], df["business_address_c"])), bool, len(df))
    return r

F = pd.concat([feats(pos, "pos"), feats(neg, "neg")], ignore_index=True)
del pos, neg; gc.collect()

# 5. the output that matters
q = [0.05, 0.25, 0.5, 0.75, 0.95]
print("\n=== NAME signal (pos vs neg) ===")
for lab in ["pos","neg"]:
    s = F[F.label==lab]
    print(f"{lab}: exact={s.name_exact.mean():.3f} | ratio q:",
          s.name_ratio.quantile(q).round(1).to_dict(), "| jacc q:", s.name_jacc.quantile(q).round(2).to_dict())

print("\n=== ADDRESS signal (pos vs neg) ===")
for lab in ["pos","neg"]:
    s = F[F.label==lab]
    print(f"{lab}: addr_missing={s.addr_miss.mean():.3f} | addr_exact={s.addr_exact.mean():.3f} | addr_jacc q:",
          s.addr_jacc.quantile(q).round(2).to_dict())

print("\n=== QUADRANTS (% of pairs; name high = ratio>=60, addr high = jacc>=0.5 & present) ===")
def quad(s):
    nh = s.name_ratio >= 60
    ah = (s.addr_jacc >= 0.5) & ~s.addr_miss
    return pd.Series({
        "1 name-high/addr-high": (nh & ah).mean(),
        "2 name-high/addr-low":  (nh & ~ah & ~s.addr_miss).mean(),
        "3 name-low/addr-high":  (~nh & ah).mean(),
        "4 name-low/addr-low":   (~nh & ~ah & ~s.addr_miss).mean(),
        "addr-missing":          s.addr_miss.mean(),
    })
print(pd.DataFrame({"pos": quad(F[F.label=="pos"]), "neg": quad(F[F.label=="neg"])}).round(3))

# --- cell 5 ---
import numpy as np, pandas as pd, re, gc
from rapidfuzz import fuzz

rng = np.random.default_rng(0)
N_POS, N_NEG = 25000, 40000
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", str(s))).upper().strip()

# identical sampling + assembly as EDA-01
m_full = gt["matched_entity_ids"].fillna("")
p = ((m_full.str.count(",")+1).where(m_full!="",0)).to_numpy(dtype=float); p /= p.sum()
samp = gt.iloc[rng.choice(len(gt), N_POS, p=p)]
ex = samp.assign(m=samp["matched_entity_ids"].fillna("").str.split(",")).explode("m")
ex = ex[ex["m"].notna() & (ex["m"] != "")]
pp = ex[["source1_entity_id","m"]].rename(columns={"source1_entity_id":"s1","m":"cand"})
pools = {}
for nm, df in [("S2",s2),("S3",s3)]:
    for c in df["country"].unique():
        pools[(nm,str(c))] = df.loc[df["country"]==c,"entity_id"].to_numpy()
i = rng.choice(len(s1), N_NEG, replace=False); s1samp = s1.iloc[i]
neg_rows = []
for eid, c in zip(s1samp["entity_id"], s1samp["country"]):
    nm = "S2" if rng.random() < 0.5 else "S3"; arr = pools[(nm,str(c))]
    neg_rows.append((eid, arr[rng.integers(len(arr))]))
pn = pd.DataFrame(neg_rows, columns=["s1","cand"])
del samp, ex, s1samp, neg_rows, p, m_full; gc.collect()

def assemble(prs):
    def pull(ids, df): return df[df["entity_id"].isin(ids)][["entity_id","business_name","business_address","country"]]
    s1t = pull(set(prs["s1"]), s1)
    ct = pd.concat([pull(set(prs.loc[prs["cand"].str[:2]=="S2","cand"]), s2),
                    pull(set(prs.loc[prs["cand"].str[:2]=="S3","cand"]), s3)])
    out = prs.merge(s1t, left_on="s1", right_on="entity_id") \
             .merge(ct, left_on="cand", right_on="entity_id", suffixes=("_1","_c"))
    return out.drop(columns=[c for c in out.columns if c.startswith("entity_id")])

pos, neg = assemble(pp), assemble(pn)
del pp, pn, pools; gc.collect()

def add_feats(df):
    n1 = [norm(x) for x in df["business_name_1"]]; n2 = [norm(x) for x in df["business_name_c"]]
    a1 = [frozenset(norm(x).split()) if isinstance(x, str) else frozenset() for x in df["business_address_1"]]
    a2 = [frozenset(norm(x).split()) if isinstance(x, str) else frozenset() for x in df["business_address_c"]]
    df = df.copy()
    df["name_ratio"] = [fuzz.ratio(a,b) for a,b in zip(n1,n2)]
    df["addr_miss"]  = [len(x)==0 or len(y)==0 for x,y in zip(a1,a2)]
    df["addr_jacc"]  = [ (len(x&y)/len(x|y)) if (x and y) else np.nan for x,y in zip(a1,a2) ]
    return df
pos, neg = add_feats(pos), add_feats(neg)

# buckets
A = pos[(pos.name_ratio<40) & ~pos.addr_miss & (pos.addr_jacc<0.4)]   # hard core
B = pos[(pos.name_ratio<40) & (pos.addr_jacc>=0.6)]                    # address-driven
C = neg[neg.name_ratio>=60]                                            # name false-friends
D = pos[(pos.name_ratio<10) & pos.addr_miss]                           # no signal at all?
print(f"% of positives : A(hard core)={len(A)/len(pos):.3f}  B(addr-driven)={len(B)/len(pos):.3f}  D(no-signal)={len(D)/len(pos):.3f}")
print(f"% of negatives : C(false friends)={len(C)/len(neg):.3f}")
print(f"A addr_jacc quantiles: {A.addr_jacc.quantile([.05,.25,.5,.75]).round(2).to_dict()}")

# digit/ZIP rescue test on bucket A
def runs(s): return set(re.findall(r"\d{4,}", str(s))) if isinstance(s, str) else set()
if len(A):
    shared = np.fromiter((bool(runs(x) & runs(y)) for x,y in zip(A.business_address_1, A.business_address_c)), bool, len(A))
    print(f"A: share a 4+ digit run (ZIP/PIN etc): {shared.mean():.3f}")
    shared4 = np.fromiter((bool(set(re.findall(r'\d+', str(x))) & set(re.findall(r'\d+', str(y))))
                           for x,y in zip(A.business_address_1, A.business_address_c)), bool, len(A))
    print(f"A: share ANY number (incl street no): {shared4.mean():.3f}")

# examples
def show(df, k, title):
    print("\n" + "="*15, title, f"(showing {min(k,len(df))} of {len(df)})", "="*15)
    for _, r in df.sample(min(k,len(df)), random_state=1).iterrows():
        print(f"[nr={r.name_ratio:.0f} aj={'NA' if pd.isna(r.addr_jacc) else format(r.addr_jacc,'.2f')}]")
        print(f"  S1: {r.business_name_1} | {r.business_address_1}")
        print(f"  C : {r.business_name_c} | {r.business_address_c}")
show(A, 6, "A — POSITIVE, low name + low address (the hard core)")
show(B, 4, "B — POSITIVE, address-driven (low name, high addr)")
show(C, 6, "C — NEGATIVE with name >= 60 (the precision trap)")
show(D, 6, "D — POSITIVE with essentially no name signal AND missing address")

# --- cell 6 ---
# EDA-03: script, containment, suffix stripping
import re, numpy as np, pandas as pd
from rapidfuzz import fuzz
norm = lambda s: re.sub(r"\s+"," ",re.sub(r"[^A-Za-z0-9 ]"," ",str(s))).upper().strip()
q = [.05,.25,.5,.75,.95]

# S1. script of each name
SCRIPTS = [("Deva",r"[\u0900-\u097F]"),("Beng",r"[\u0980-\u09FF]"),("Guru",r"[\u0A00-\u0A7F]"),
           ("Gujr",r"[\u0A80-\u0AFF]"),("Taml",r"[\u0B80-\u0BFF]"),("Knda",r"[\u0C80-\u0CFF]"),
           ("Mlym",r"[\u0D00-\u0D7F]"),("Thai",r"[\u0E00-\u0E7F]"),("Cyrl",r"[\u0400-\u04FF]"),
           ("Arab",r"[\u0600-\u06FF]"),("Grek",r"[\u0370-\u03FF]"),("Hani",r"[\u4E00-\u9FFF]"),
           ("Jpan",r"[\u3040-\u30FF]"),("Kore",r"[\uAC00-\uD7AF]")]
def script(s):
    s = str(s)
    for nm, pat in SCRIPTS:
        if re.search(pat, s): return nm
    return "Latn" if re.search(r"[A-Za-z]", s) else "None"
for df in (pos, neg):
    df["sc1"] = [script(x) for x in df.business_name_1]
    df["sc2"] = [script(x) for x in df.business_name_c]

print("=== S1: SCRIPT ===")
for lab, df in [("pos",pos),("neg",neg)]:
    print(f"{lab}: cross-script rate = {(df.sc1!=df.sc2).mean():.3f} | sc1 dist:",
          df.sc1.value_counts(normalize=True).head(6).round(3).to_dict())
pc = pos[pos.sc1!=pos.sc2]
print("pos cross-script by country:", pos.assign(x=pos.sc1!=pos.sc2).groupby("country_1")["x"].mean().round(3).to_dict())
print("pos cross-script pairs: addr_jacc q:", pc.addr_jacc.quantile([.25,.5,.75]).round(2).to_dict(),
      "| addr_miss:", round(pc.addr_miss.mean(),3))

# S2. name separation WITHIN same script
print("\n=== S2: name_ratio on SAME-script pairs only ===")
for lab, df in [("pos",pos),("neg",neg)]:
    s = df[df.sc1==df.sc2]
    print(f"{lab}: n={len(s)} | ratio q:", s.name_ratio.quantile(q).round(1).to_dict())

# S3. address containment (fragment-friendly) vs jaccard
print("\n=== S3: address containment |a∩b|/min(|a|,|b|) vs jaccard ===")
def toks(x): return frozenset(norm(x).split()) if isinstance(x, str) else frozenset()
def cont(x, y):
    a, b = toks(x), toks(y)
    if not a or not b: return np.nan
    return len(a & b) / min(len(a), len(b))
for lab, df in [("pos",pos),("neg",neg)]:
    c = pd.Series([cont(x,y) for x,y in zip(df.business_address_1, df.business_address_c)], dtype=float)
    print(f"{lab}: containment q:", c.quantile(q).round(2).to_dict(),
          "| >=0.9:", round((c>=0.9).mean(),3), "| ==1.0:", round((c==1.0).mean(),3))

def nums(s): return set(re.findall(r"\d+", str(s))) if re.search(r"\d", str(s)) else set()
for lab, df in [("pos",pos),("neg",neg)]:
    sh = np.fromiter((bool(nums(x)&nums(y)) for x,y in zip(df.business_address_1, df.business_address_c)), bool, len(df))
    print(f"{lab}: share ANY number = {sh.mean():.3f}")

# S4. suffix-stripped name similarity
print("\n=== S4: strip legal suffixes, recompute ratio ===")
SUF = r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY)\b"
def strip_suf(s): return re.sub(r"\s+"," ", re.sub(SUF," ", norm(s))).strip()
for lab, df in [("pos",pos),("neg",neg)]:
    r0, r1 = [], []
    for x,y in zip(df.business_name_1, df.business_name_c):
        nx, ny = norm(x), norm(y)
        r0.append(fuzz.ratio(nx, ny))
        r1.append(fuzz.ratio(strip_suf(x), strip_suf(y)))
    r0, r1 = pd.Series(r0), pd.Series(r1)
    print(f"{lab}: raw q:", r0.quantile(q).round(1).to_dict())
    print(f"{lab}: stripped q:", r1.quantile(q).round(1).to_dict(),
          f"| % pairs that were >=60 raw but <60 stripped: {((r0>=60)&(r1<60)).mean():.3f}")
neg["raw"], neg["stripped"] = r0, r1   # keep for last line
print("neg false-friend rate: raw>=60:", round((neg.raw>=60).mean(),3),
      "| stripped>=60:", round((neg.stripped>=60).mean(),3))

# --- cell 7 ---
# EDA-04: rule coverage + residual anatomy
import re, numpy as np, pandas as pd
from rapidfuzz import fuzz
norm = lambda s: re.sub(r"\s+"," ",re.sub(r"[^A-Za-z0-9 ]"," ",str(s))).upper().strip()
SUF = r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY)\b"
strip_suf = lambda s: re.sub(r"\s+"," ", re.sub(SUF," ", norm(s))).strip()
def toks(x): return frozenset(norm(x).split()) if isinstance(x, str) else frozenset()
def nums(x): return set(re.findall(r"\d+", str(x))) if re.search(r"\d", str(x)) else set()

for df in (pos, neg):
    df["nr_s"]   = [fuzz.ratio(strip_suf(x), strip_suf(y))
                    for x,y in zip(df.business_name_1, df.business_name_c)]
    a1 = [toks(x) for x in df.business_address_1]; a2 = [toks(y) for y in df.business_address_c]
    df["cont"]   = [(len(a&b)/min(len(a),len(b))) if (a and b) else np.nan
                    for a,b in zip(a1,a2)]
    df["shnum"]  = [bool(nums(x)&nums(y)) for x,y in zip(df.business_address_1, df.business_address_c)]
    df["same_sc"]= df.sc1 == df.sc2
    df["R1"]  = df.same_sc & (df.nr_s >= 60)      # name rule
    df["R2"]  = df.cont >= 0.9                    # address containment rule
    df["R3"]  = df.shnum                          # shared number rule
    df["U12"]  = df.R1 | df.R2
    df["U123"] = df.R1 | df.R2 | df.R3

print("=== coverage(pos) vs rate(neg) ===")
tbl = pd.DataFrame({r: {"pos_cov": pos[r].mean(), "neg_rate": neg[r].mean()}
                    for r in ["R1","R2","R3","U12","U123"]}).T
print(tbl.round(3))
print("\npos U123 by country:", pos.groupby("country_1")["U123"].mean().round(3).to_dict())
print("pos U123: cross-script:", round(pos[pos.sc1!=pos.sc2]["U123"].mean(),3),
      "| same-script:", round(pos[pos.sc1==pos.sc2]["U123"].mean(),3))

# the residual: positives NO rule catches
res = pos[~pos.U123]
print(f"\nRESIDUAL positives: {len(res)} ({len(res)/len(pos):.3f}) | cross-script share: {(res.sc1!=res.sc2).mean():.3f}")
def show(df, k, title):
    print("\n" + "="*12, title, "="*12)
    if len(df) == 0: print("(empty)"); return
    for _, r in df.sample(min(k,len(df)), random_state=1).iterrows():
        print(f"[nr_s={r.nr_s:.0f} cont={'NA' if pd.isna(r.cont) else format(r.cont,'.2f')} "
              f"shnum={r.shnum} cross_sc={r.sc1!=r.sc2}]")
        print(f"  S1: {r.business_name_1} | {r.business_address_1}")
        print(f"  C : {r.business_name_c} | {r.business_address_c}")
show(res, 10, "RESIDUAL — positives no simple rule catches")

# where do false positives come from?
neg_only3 = neg[neg.R3 & ~neg.R1 & ~neg.R2]
print(f"\nneg caught ONLY by shared-number: {len(neg_only3)/len(neg):.3f}")
show(neg_only3, 4, "NEG — shared number only (is it coincidence or near-dup?)")

# --- cell 8 ---
# EDA-05: structure, overlap, key selectivity
import pandas as pd, numpy as np, gc
SUF = r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY)\b"
def vnorm(col):   # vectorized: uppercase, alnum-only, collapse space, strip legal suffix
    return (col.str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
               .str.replace(SUF, " ", regex=True)
               .str.replace(r"\s+", " ", regex=True).str.strip())

# S1. uniqueness & exact duplicates
print("=== S1: uniqueness & exact duplicates ===")
for n, d in [("s1", s1), ("s2", s2), ("s3", s3)]:
    dup = d.duplicated(subset=["business_name","business_address","country"], keep=False).sum()
    print(f"{n}: id_unique={d.entity_id.is_unique} | rows in exact-dup groups: {dup:,} ({dup/len(d):.3%})")

# S2. exact-name key selectivity on the pool
print("\n=== S2: (country, exact-stripped-name) key selectivity ===")
pk = pd.concat([s2[["country","business_name"]], s3[["country","business_name"]]], ignore_index=True)
key = pk["country"] + "|" + vnorm(pk["business_name"])
del pk; gc.collect()
vc = key.value_counts()
print(f"pool rows: {len(key):,} | distinct keys: {len(vc):,}")
print("bucket-size of keys:", {"size1": round((vc==1).mean(),3),
      "2-5": round(((vc>=2)&(vc<=5)).mean(),3), "6-50": round(((vc>=6)&(vc<=50)).mean(),3),
      ">50": round((vc>50).mean(),3)})
print(f"pool rows sitting in buckets >50: {vc[vc>50].sum()/len(key):.3%}")
print("largest buckets:", vc.head(6).to_dict())
s1key = s1["country"] + "|" + vnorm(s1["business_name"])
bs = s1key.map(vc)
print(f"S1 having this key in pool: {bs.notna().mean():.3f} | bucket size for those:",
      bs.dropna().quantile([.5,.9,.99]).round(0).to_dict(), "| max:", bs.max())
del key, vc, s1key, bs; gc.collect()

# S3. S2 vs S3 overlap (hash-based, memory-cheap)
print("\n=== S3: cross-source overlap ===")
hh = lambda c1, c2: pd.util.hash_pandas_object(c1 + "|" + c2, index=False).to_numpy(np.uint64)
k2 = hh(s2["country"], vnorm(s2["business_name"])); k3 = hh(s3["country"], vnorm(s3["business_name"]))
print(f"S2 keys also in S3: {np.isin(k2, k3).mean():.3f} | S3 keys also in S2: {np.isin(k3, k2).mean():.3f}")
r2 = hh(vnorm(s2["business_name"]), s2["business_address"].fillna(""))
r3 = hh(vnorm(s3["business_name"]), s3["business_address"].fillna(""))
print(f"S2 full-records (name+addr) also in S3: {np.isin(r2, r3).mean():.3f}")
k1 = hh(s1["country"], vnorm(s1["business_name"]))
print(f"S1 exact keys present in S2∪S3: {np.isin(k1, np.concatenate([k2, k3])).mean():.3f}")
del k1, k2, k3, r2, r3; gc.collect()

# S4. within-S1 match diversity (3000-row sample)
m = gt["matched_entity_ids"].fillna("")
n = (m.str.count(",") + 1).where(m != "", 0)
samp = gt[n >= 2].sample(3000, random_state=0)
ex = samp.assign(mm=samp["matched_entity_ids"].str.split(",")).explode("mm")
ex = ex[ex["mm"].notna()][["source1_entity_id", "mm"]].rename(columns={"mm": "entity_id"})

def pull(ids, d): return d[d["entity_id"].isin(ids)][["entity_id","business_name","business_address","country"]]
m2 = set(ex.loc[ex.entity_id.str[:2] == "S2", "entity_id"])
m3 = set(ex.loc[ex.entity_id.str[:2] == "S3", "entity_id"])
cand = pd.concat([pull(m2, s2), pull(m3, s3)])
s1t = pull(set(ex["source1_entity_id"]), s1).set_index("entity_id")

cand["nn"]  = vnorm(cand["business_name"])          # vnorm from the EDA-05 cell
cand["src"] = cand["entity_id"].str[:2]
merged = ex.merge(cand[["entity_id","business_name","nn","business_address","src"]], on="entity_id")

agg = merged.groupby("source1_entity_id").agg(
    n=("nn","size"),
    distinct_names=("nn","nunique"),
    n_s2=("src", lambda s: (s=="S2").sum()),
    n_s3=("src", lambda s: (s=="S3").sum()))

print("per-S1: matches n:", agg.n.quantile([.5,.9]).to_dict(),
      "| DISTINCT names among its matches:", agg.distinct_names.quantile([.5,.9,.99]).to_dict(),
      "| max:", agg.distinct_names.max())
print(f"S1 whose every match shares ONE exact name: {(agg.distinct_names==1).mean():.3f}")
print(f"S1 with >=3 distinct names among matches:  {(agg.distinct_names>=3).mean():.3f}")

big = list(agg[agg.distinct_names >= 4].index[:3])
print("diverse groups found:", len(big))
for sid in big:
    a = s1t.loc[sid]
    print(f"\nS1 {sid}: {a.business_name} | {a.business_address}")
    for _, r in merged[merged["source1_entity_id"] == sid].iterrows():
        print(f"   {r.entity_id[:2]}: {r.business_name} | {r.business_address}")

# --- cell 9 ---
import pandas as pd
BASE = "/home/ubuntu/dataset/student_resource/dataset"
t1 = pd.read_csv(f"{BASE}/test/test_source1.tsv", sep="\t")
t2 = pd.read_csv(f"{BASE}/test/test_source2.tsv", sep="\t")
t3 = pd.read_csv(f"{BASE}/test/test_source3.tsv", sep="\t")

# A. shapes, countries, nulls
print("=== A. test structure ===")
for n, d in [("t1", t1), ("t2", t2), ("t3", t3)]:
    print(n, d.shape, list(d.columns), "| countries:", d["country"].value_counts(dropna=False).to_dict())
print("nulls:", {n: d.isna().mean().round(3).to_dict() for n, d in [("t1",t1),("t2",t2),("t3",t3)]})

# B. normalization sanity: OLD (script-erasing) vs NEW (ASCII-punct-only)
PUNCT = r"[!-/:-@\[-`{-~]"          # all ASCII punctuation ranges
def n_old(c): return (c.fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
                           .str.replace(r"\s+", " ", regex=True).str.strip())
def n_new(c): return (c.fillna("").str.upper().str.replace(PUNCT, " ", regex=True)
                           .str.replace(r"\s+", " ", regex=True).str.strip())
print("\n=== B. names normalizing to EMPTY (the India| bug) ===")
for n, d in [("s1",s1), ("s2",s2), ("s3",s3), ("t1",t1), ("t2",t2), ("t3",t3)]:
    o = n_old(d["business_name"]) == ""
    print(f"{n}: OLD-empty={o.mean():.4f}", end=" | ")
    print(f"NEW-empty={(n_new(d['business_name'])=='').mean():.4f}")
for n, d in [("s2",s2), ("s3",s3), ("t2",t2), ("t3",t3)]:
    e = n_old(d["business_name"]) == ""
    print(f"{n} OLD-empty by country:", d.assign(e=e).groupby("country")["e"].mean().round(4).to_dict())

# C. non-ASCII names (accents / scripts) by country
print("\n=== C. names containing non-ASCII chars ===")
for n, d in [("s1",s1), ("s2",s2), ("t1",t1), ("t2",t2), ("t3",t3)]:
    na = d["business_name"].fillna("").str.contains(r"[^\x00-\x7F]", regex=True)
    print(n, d.assign(na=na).groupby("country")["na"].mean().round(4).to_dict())

# D. France address & suffix structure
print("\n=== D. France structure ===")
fr_name = [c for c in t2["country"].unique() if str(c).strip().lower().startswith("fr")]
print("France country value(s) in test:", fr_name)
if fr_name:
    fr = t2[t2["country"].isin(fr_name)]
    addr = fr["business_address"].fillna("")
    print("t2 France rows:", len(fr))
    print("5-digit postal present:", round(addr.str.contains(r"\b\d{5}\b", regex=True).mean(), 4),
          "| CEDEX:", round(addr.str.contains("CEDEX", case=False, regex=False).mean(), 4))
    for suf in ["SAS", "SARL", "SASU", "EURL", "SCI", "SA", "EIRL"]:
        r = fr["business_name"].str.upper().str.contains(rf"\b{suf}\b", regex=True).mean()
        if r > 0: print(f"  suffix {suf}: {r:.4f}")
    print("France name samples:")
    print(fr.sample(6, random_state=0)[["business_name","business_address"]].to_string(index=False))
# compare: US/India suffix rates for scale
for suf in ["LLC","INC","PVT","PRIVATE"]:
    r1 = t1[t1.country=="US"]["business_name"].str.upper().str.contains(rf"\b{suf}\b", regex=True).mean() if (t1.country=="US").any() else float("nan")
    print(f"t1 US suffix {suf}: {r1:.4f}")
del t1, t2, t3, fr, addr; gc.collect()

# --- cell 10 ---
# EDA-07: field internals — numbers, postal keys, char inventory
import pandas as pd, numpy as np, re, gc, unicodedata
from collections import Counter

def profile(d, label):
    a  = d["business_address"].fillna("")
    nm = d["business_name"].fillna("")
    s_any  = a.str.contains(r"\d", regex=True)
    s_lead = a.str.match(r"^\s*\d")
    s_5    = a.str.contains(r"\b\d{5}\b", regex=True)
    s_6    = a.str.contains(r"\b\d{6}\b", regex=True)
    s_nd   = nm.str.contains(r"\d", regex=True)
    print(f"--- {label} ---")
    for col, s in [("addr_any_digit", s_any), ("addr_leading_number", s_lead),
                   ("addr_5digit", s_5), ("addr_6digit", s_6), ("name_has_digit", s_nd)]:
        print(f"  {col}: overall={s.mean():.4f} | by-country={d.assign(x=s).groupby('country')['x'].mean().round(3).to_dict()}")
    del a, nm, s_any, s_lead, s_5, s_6, s_nd; gc.collect()

for lbl, d in [("s1", s1), ("s2", s2), ("s3", s3)]:
    profile(d, lbl)

# France side: re-read only t1 (1.7M — small), profile, delete
BASE = "/home/ubuntu/dataset/student_resource/dataset"
t1 = pd.read_csv(f"{BASE}/test/test_source1.tsv", sep="\t")
profile(t1, "t1 (test S1)")
del t1; gc.collect()

# char inventory: WHICH non-ASCII chars, and what categories
def char_inventory(d, label):
    print(f"=== char inventory: {label} ===")
    for c in d["country"].unique():
        sub = d.loc[d["country"] == c, "business_name"].fillna("")
        na = sub[sub.str.contains(r"[^\x00-\x7F]", regex=True)]
        if len(na) == 0:
            print(f"  {c}: no non-ASCII"); continue
        samp = na.sample(min(5000, len(na)), random_state=0)
        chars = [ch for s in samp for ch in s if ord(ch) > 127]
        cats = Counter(unicodedata.category(ch)[0] for ch in chars)   # L=letter M=mark P=punct S=symbol
        print(f"  {c}: names_with_nonASCII={len(na):,} | categories={dict(cats)} | top={Counter(chars).most_common(15)}")

char_inventory(s1, "train s1")
char_inventory(s2, "train s2 (sample of 5000 names/country)")
t1 = pd.read_csv(f"{BASE}/test/test_source1.tsv", sep="\t")
char_inventory(t1, "test t1")
del t1; gc.collect()

# --- cell 11 ---
import numpy as np, pandas as pd, re, gc

rng = np.random.default_rng(0)
N_POS, N_NEG = 25000, 40000                      # SAMPLE-BASED
PUNCT = r"[!-/:-@\[-`{-~]"
SUF   = r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY)\b"

def nbase(s):
    return re.sub(r"\s+", " ", re.sub(PUNCT, " ", str(s).upper())).strip()
def nstrip(s):
    return re.sub(r"\s+", " ", re.sub(SUF, " ", nbase(s))).strip()
def toks(s):  return frozenset(t for t in nbase(s).split() if len(t) >= 3)
def nums(s, ml=1): return frozenset(x for x in re.findall(r"\d+", str(s)) if len(x) >= ml)
def nonlatin(s): return bool(re.search(r"[\u0900-\u0D7F\u0A00-\u0A7F\u0980-\u09FF]", str(s)))

# rebuild deterministic pairs (same scheme as EDA-01)
m_full = gt["matched_entity_ids"].fillna("")
p = ((m_full.str.count(",")+1).where(m_full!="",0)).to_numpy(dtype=float); p /= p.sum()
samp = gt.iloc[rng.choice(len(gt), N_POS, p=p)]
ex = samp.assign(m=samp["matched_entity_ids"].str.split(",")).explode("m")
ex = ex[ex["m"].notna() & (ex["m"]!="")]
pp = ex[["source1_entity_id","m"]].rename(columns={"source1_entity_id":"s1","m":"cand"})
pools = {}
for nm, df in [("S2",s2),("S3",s3)]:
    for c in df["country"].unique():
        pools[(nm,str(c))] = df.loc[df["country"]==c,"entity_id"].to_numpy()
i = rng.choice(len(s1), N_NEG, replace=False); s1samp = s1.iloc[i]
rows = []
for eid, c in zip(s1samp["entity_id"], s1samp["country"]):
    nm = "S2" if rng.random() < 0.5 else "S3"; arr = pools[(nm,str(c))]
    rows.append((eid, arr[rng.integers(len(arr))]))
pn = pd.DataFrame(rows, columns=["s1","cand"])
del samp, ex, s1samp, rows, p, m_full, pools; gc.collect()

def assemble(prs):
    def pull(ids, d): return d[d["entity_id"].isin(ids)][["entity_id","business_name","business_address","country"]]
    s1t = pull(set(prs["s1"]), s1)
    ct = pd.concat([pull(set(prs.loc[prs.cand.str[:2]=="S2","cand"]), s2),
                    pull(set(prs.loc[prs.cand.str[:2]=="S3","cand"]), s3)])
    out = prs.merge(s1t, left_on="s1", right_on="entity_id") \
             .merge(ct, left_on="cand", right_on="entity_id", suffixes=("_1","_c"))
    return out.drop(columns=[c for c in out.columns if c.startswith("entity_id")])

pos, neg = assemble(pp), assemble(pn)
del pp, pn; gc.collect()

def keymask(df):
    n1  = [nbase(x)  for x in df.business_name_1];  n2  = [nbase(x)  for x in df.business_name_c]
    s1_ = [nstrip(x) for x in df.business_name_1];  s2_ = [nstrip(x) for x in df.business_name_c]
    t1_ = [toks(x)   for x in df.business_name_1];  t2_ = [toks(x)   for x in df.business_name_c]
    a1  = [nums(x,1) for x in df.business_address_1.fillna("")]; a2 = [nums(x,1) for x in df.business_address_1.fillna("")]
    a2  = [nums(x,1) for x in df.business_address_c.fillna("")]
    b1  = [nums(x,2) for x in df.business_address_1.fillna("")]; b2 = [nums(x,2) for x in df.business_address_c.fillna("")]
    M = pd.DataFrame(index=df.index)
    M["exact"] = np.fromiter((a==b for a,b in zip(n1,n2)),  bool, len(df))
    M["strip"] = np.fromiter((a==b for a,b in zip(s1_,s2_)), bool, len(df))
    M["pref4"] = np.fromiter((len(a)>0 and a[:4]==b[:4] for a,b in zip(n1,n2)), bool, len(df))
    M["token"] = np.fromiter((bool(a&b) for a,b in zip(t1_,t2_)), bool, len(df))
    M["num1"]  = np.fromiter((bool(a&b) for a,b in zip(a1,a2)),   bool, len(df))
    M["num2"]  = np.fromiter((bool(a&b) for a,b in zip(b1,b2)),   bool, len(df))
    M["U1_strip_token_num2"] = M.strip | M.token | M.num2
    M["U2_exact_token_num1"] = M.exact | M.token | M.num1
    return M

mp, mn = keymask(pos), keymask(neg)
cross = np.fromiter((nonlatin(x) for x in pos.business_name_c), bool, len(pos))
us, ind = (pos.country_1=="US").to_numpy(), (pos.country_1=="India").to_numpy()

print(f"{'key':24s} {'pos':>7s} {'neg':>7s} | {'posUS':>7s} {'posInd':>7s} {'cross-sc':>8s} {'same-sc':>8s}")
for k in mp.columns:
    print(f"{k:24s} {mp[k].mean():7.3f} {mn[k].mean():7.3f} | "
          f"{mp[k].to_numpy()[us].mean():7.3f} {mp[k].to_numpy()[ind].mean():7.3f} "
          f"{mp[k].to_numpy()[cross].mean():8.3f} {mp[k].to_numpy()[~cross].mean():8.3f}")

# pairs caught by NEITHER union
miss = ~(mp.U1_strip_token_num2 | mp.U2_exact_token_num1)
print(f"\nmissed by both unions: {miss.sum()} ({miss.mean():.4f})")
mpx = pos[miss.values]
for _, r in mpx.sample(min(5, len(mpx)), random_state=1).iterrows():
    print(f"  S1: {r.business_name_1} | {r.business_address_1}")
    print(f"  C : {r.business_name_c} | {r.business_address_c}")

# --- cell 12 ---
# EDA-09: blocking-key SELECTIVITY (full pool, polars streaming)
# [notebook shell] !pip install -q polars==1.41.2
import polars as pl, numpy as np, gc
BASE = "/home/ubuntu/dataset/student_resource/dataset"
PUNCT = r"[!-/:-@\[-`{-~]"
SUF   = r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY)\b"
CAP_NAME, CAP_ADDR, CAP_NUM = 300, 300, 100

def nn(c):
    return c.str.to_uppercase().str.replace_all(PUNCT, " ").str.replace_all(r"\s+", " ").str.strip_chars()
def nns(c):
    return nn(c).str.replace_all(SUF, " ").str.replace_all(r"\s+", " ").str.strip_chars()

s1p = pl.read_csv(f"{BASE}/train/train_source1.tsv", separator="\t")   # eager DataFrame
lf2 = pl.scan_csv(f"{BASE}/train/train_source2.tsv", separator="\t")   # LazyFrame
lf3 = pl.scan_csv(f"{BASE}/train/train_source3.tsv", separator="\t")
POOL = 10_320_219

# A. exact / strip / prefix4 bucket sizes (norm2)
def bucket_stats(label, fn):
    def kc(lf):
        return lf.select([(pl.col("country") + "|" + fn(pl.col("business_name"))).alias("k")]) \
                 .group_by("k").len().collect()
    a, b = kc(lf2), kc(lf3)                      # DataFrames
    tot = a.join(b, on="k", how="full", suffix="_b") \
           .with_columns((pl.col("len") + pl.col("len_b").fill_null(0)).alias("size")) \
           .drop("len", "len_b")                 # DataFrame
    s1k = s1p.select([(pl.col("country") + "|" + fn(pl.col("business_name"))).alias("k")])
    sizes = s1k.join(tot, on="k", how="left")["size"].fill_null(0).to_numpy()   # no .collect() (fixed upstream)
    big = tot.filter(pl.col("size") > 50)["size"].sum() / POOL
    top = tot.top_k(5, by="size").rows()
    print(f"[{label}] S1 key present={np.mean(sizes>0):.3f} | bucket: med={np.median(sizes):.0f} "
          f"p90={np.percentile(sizes,90):.0f} p99={np.percentile(sizes,99):.0f} max={sizes.max()} "
          f"| pool rows in buckets>50: {big:.3%}")
    print(f"    est pairs from this key: {sizes.sum():,} | largest buckets: {top[:3]}")
    del a, b, tot, s1k, sizes; gc.collect()

print("=== A. name-key bucket selectivity (norm2) ===")
bucket_stats("exact",  nn)
bucket_stats("strip",  nns)
bucket_stats("pref4",  lambda c: nn(c).str.slice(0, 4))

# B. token / number document frequencies (streaming)
def tok_df(lf, col):
    return (lf.select(nn(pl.col(col)).str.split(" ").explode().alias("t"))
               .filter(pl.col("t").str.len_chars() >= 3)
               .group_by("t").len().rename({"len": "df"})
               .collect(engine="streaming"))
def num_df(lf):
    return (lf.select(pl.col("business_address").fill_null("")
                        .str.extract_all(r"\d+").explode().alias("n"))
               .filter(pl.col("n").str.len_chars() >= 2)
               .group_by("n").len().rename({"len": "df"})
               .collect(engine="streaming"))

print("\n=== B. document frequencies (streaming — this section takes minutes) ===")
ntok = tok_df(lf2, "business_name").join(tok_df(lf3, "business_name"), on="t", how="full", suffix="_b") \
       .with_columns((pl.col("df") + pl.col("df_b").fill_null(0)).alias("df")).drop("df_b")
atok = tok_df(lf2, "business_address").join(tok_df(lf3, "business_address"), on="t", how="full", suffix="_b") \
       .with_columns((pl.col("df") + pl.col("df_b").fill_null(0)).alias("df")).drop("df_b")
nnum = num_df(lf2).join(num_df(lf3), on="n", how="full", suffix="_b") \
       .with_columns((pl.col("df") + pl.col("df_b").fill_null(0)).alias("df")).drop("df_b")
print(f"vocab: name-tokens={len(ntok):,} (rare<= {CAP_NAME}: {(ntok['df']<=CAP_NAME).sum():,}) | "
      f"addr-tokens={len(atok):,} (rare<= {CAP_ADDR}: {(atok['df']<=CAP_ADDR).sum():,}) | "
      f"numbers>=2dig={len(nnum):,} (rare<= {CAP_NUM}: {(nnum['df']<=CAP_NUM).sum():,})")

print("\n=== C. candidates/S1 estimate (sum-DF upper bound) ===")
def est(s1_tokens, counts, cap, label):
    usable = counts.filter((pl.col("df") >= 1) & (pl.col("df") <= cap))
    e = s1_tokens.join(usable, on="t").group_by("i").agg(pl.col("df").sum())
    arr = np.zeros(s1p.height, dtype=np.int64)
    arr[e["i"].to_numpy()] = e["df"].to_numpy()
    print(f"{label}: mean={arr.mean():.1f} med={np.median(arr):.0f} p90={np.percentile(arr,90):.0f} "
          f"p99={np.percentile(arr,99):.0f} max={arr.max()} | S1 with 0={np.mean(arr==0):.4f} "
          f"| total={arr.sum():,}")
    return arr

s1n = (s1p.with_row_index("i")
           .with_columns(nn(pl.col("business_name")).str.split(" ").alias("tl"))
           .explode("tl").select(pl.col("i"), pl.col("tl").alias("t")))
s1a = (s1p.with_row_index("i")
           .with_columns(nn(pl.col("business_address").fill_null("")).str.split(" ").alias("tl"))
           .explode("tl").select(pl.col("i"), pl.col("tl").alias("t")))
s1num = (s1p.with_row_index("i")
             .with_columns(pl.col("business_address").fill_null("").str.extract_all(r"\d+").alias("tl"))
             .explode("tl").select(pl.col("i"), pl.col("tl").alias("t")))

e1 = est(s1n, ntok, CAP_NAME, "name-token (cap 300)")
e2 = est(s1a, atok, CAP_ADDR, "addr-token (cap 300)")
nnum_t = nnum.rename({"n": "t"})
e3 = est(s1num, nnum_t, CAP_NUM, "number>=2d (cap 100)")
tot_ub = e1 + e2 + e3
print(f"SUM of all three (union upper bound): mean={tot_ub.mean():.1f} p99={np.percentile(tot_ub,99):.0f}")
print("NOTE: union <= sum (overlap not removed). Exact union measured on sample next.")

# --- cell 13 ---
# [notebook magic] %%time
import gc, unicodedata
import numpy as np
import polars as pl
from pathlib import Path

# paths
TRAIN = Path("/home/ubuntu/dataset/student_resource/dataset/train")
def find(name):
    p = TRAIN / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def pick(cols, *c):
    for x in c:
        if x in cols: return x
    raise KeyError(f"none of {c} in {list(cols)}")

def cols(df):
    return dict(id=pick(df.columns,"entity_id","id"),
                name=pick(df.columns,"business_name","name"),
                addr=pick(df.columns,"business_address","address"),
                country=pick(df.columns,"country"))

# norm2 (convention: uppercase -> strip ASCII punct -> collapse ws)
def N(e):
    return (e.str.to_uppercase()
             .str.replace_all(r"[!-/:-@\[-`{-~]", "")
             .str.replace_all(r"\s+", " ")
             .str.strip_chars())

def fold(s):   # accent-fold for the extra exact key (Latin only use-case)
    if s is None: return None
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(ch))

def load(name):
    df = pl.read_csv(find(name), separator="\t")
    i = pick(df.columns, "id", "entity_id", "source_id", "source1_entity_id")
    return df.with_columns(pl.col(i).cast(pl.Utf8))

s1 = load("train_source1.tsv"); k1 = cols(s1)
s2 = load("train_source2.tsv"); k2 = cols(s2)
s3 = load("train_source3.tsv"); k3 = cols(s3)
gt = load("train_ground_truth.tsv")
GID = pick(gt.columns, "source1_entity_id", "s1_id", "id")
GM  = pick(gt.columns, "matched_entity_ids", "matches", "matched_ids")
print("shapes:", s1.shape, s2.shape, s3.shape, gt.shape)

# GT: derive n_matches locally, auto-detect separator
raw = gt[GM].fill_null("")
seps = [",", ";", "|", " "]
sep = max(seps, key=lambda s: raw.str.contains(s, literal=True).fill_null(False).sum())
print("GT separator:", repr(sep))
mlist = raw.str.split(sep)
gt2 = gt.with_columns(mlist.alias("m")).with_columns(pl.col("m").list.len().alias("n_matches"))
print("mean n_matches (expect ~3.46):", round(gt2["n_matches"].mean(), 4),
      "| singles:", int((gt2["n_matches"] == 1).sum()))

# pos: 25k GT rows weighted by n_matches, exploded
rng = np.random.default_rng(0)
w = gt2["n_matches"].to_numpy().astype("float64"); w /= w.sum()
idx = rng.choice(gt2.height, size=25_000, replace=False, p=w).tolist()
pos = (gt2.with_row_index("ri").filter(pl.col("ri").is_in(idx))
          .explode("m")
          .select([pl.col(GID).alias("s1_id"), pl.col("m").alias("cand_id"),
                   pl.lit(1).alias("y")])
          .with_columns(pl.col("cand_id").str.strip_chars().alias("cand_id"))
          .filter((pl.col("cand_id") != "") & pl.col("cand_id").is_not_null()))

# neg: 40k same-country random pairs
pool = pl.concat([
    s2.select([pl.col(k2["id"]).alias("pid"), pl.col(k2["country"]).alias("ct")]),
    s3.select([pl.col(k3["id"]).alias("pid"), pl.col(k3["country"]).alias("ct")]),
])
sidx = rng.choice(s1.height, size=40_000, replace=False).tolist()
s1samp = s1.with_row_index("ri").filter(pl.col("ri").is_in(sidx))
neg_frames, skipped = [], 0
for i, (key, sub) in enumerate(s1samp.group_by(k1["country"])):
    ct = key[0]
    cand = pool.filter(pl.col("ct") == ct)
    if cand.height == 0:
        skipped += len(sub); continue
    take = cand.sample(n=len(sub), with_replacement=True, seed=100 + i)
    neg_frames.append(pl.DataFrame({
        "s1_id": sub[k1["id"]].to_numpy(),
        "cand_id": take["pid"].to_numpy(),
        "y": np.zeros(len(sub), dtype=np.int32)}))
neg = pl.concat(neg_frames)
print("neg pairs:", neg.height, "| skipped (no pool country):", skipped)

pairs = pl.concat([pos, neg]).with_row_index("pair_i")
print("pairs:", pairs.height, "| pos:", int((pairs["y"]==1).sum()),
      "| neg:", int((pairs["y"]==0).sum()))

# pull text via isin (derived vars only; s1/s2/s3 untouched)
all_s1 = pairs["s1_id"].unique()
all_c  = pairs["cand_id"].unique()
s1x = s1.filter(pl.col(k1["id"]).is_in(all_s1)).select([
    pl.col(k1["id"]).alias("s1_id"), pl.col(k1["name"]).alias("s1_name"),
    pl.col(k1["addr"]).alias("s1_addr"), pl.col(k1["country"]).alias("s1_country")])
candx = pl.concat([
    s2.filter(pl.col(k2["id"]).is_in(all_c)).select([
        pl.col(k2["id"]).alias("cid"), pl.col(k2["name"]).alias("c_name"),
        pl.col(k2["addr"]).alias("c_addr"), pl.col(k2["country"]).alias("c_country")]),
    s3.filter(pl.col(k3["id"]).is_in(all_c)).select([
        pl.col(k3["id"]).alias("cid"), pl.col(k3["name"]).alias("c_name"),
        pl.col(k3["addr"]).alias("c_addr"), pl.col(k3["country"]).alias("c_country")]),
])
feat = (pairs.join(s1x, on="s1_id", how="left")
              .join(candx, left_on="cand_id", right_on="cid", how="left"))
print("pull nulls (expect 0,0):", feat["s1_name"].null_count(), feat["c_name"].null_count())
print("countries:", {r[0]: r[1] for r in feat.group_by("s1_country").len().rows()})

# per-pair keys
DEV = r"[\u0900-\u097F]"
feat = (feat
    .with_columns([
        N(pl.col("s1_name")).alias("sn1"), N(pl.col("c_name")).alias("sn2"),
        N(pl.col("s1_addr")).alias("sa1"), N(pl.col("c_addr")).alias("sa2"),
        pl.col("s1_name").str.contains(DEV).fill_null(False).alias("d1"),
        pl.col("c_name").str.contains(DEV).fill_null(False).alias("d2"),
    ])
    .with_columns([
        pl.col("sn1").str.extract_all(r"\w+").list.unique().alias("n1"),
        pl.col("sn2").str.extract_all(r"\w+").list.unique().alias("n2"),
        pl.col("sa1").str.extract_all(r"\w+").list.unique().alias("a1"),
        pl.col("sa2").str.extract_all(r"\w+").list.unique().alias("a2"),
        pl.col("sa1").str.extract_all(r"\d{2,}").list.unique().alias("m1"),
        pl.col("sa2").str.extract_all(r"\d{2,}").list.unique().alias("m2"),
        ((pl.col("sn1") == pl.col("sn2")) & (pl.col("sn1").str.len_chars() > 0)).alias("exact"),
        pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("f1"),
        pl.col("sn2").map_elements(fold, return_dtype=pl.Utf8).alias("f2"),
    ])
    .with_columns([
        ((pl.col("f1") == pl.col("f2")) & (pl.col("f1").str.len_chars() > 0)).fill_null(False).alias("exact_fold"),
        (pl.col("d1") != pl.col("d2")).alias("cross"),
    ]))

def shared(df, c1, c2, tag):
    l = df.select(["pair_i", pl.col(c1).alias("t")]).explode("t").drop_nulls()
    r = df.select(["pair_i", pl.col(c2).alias("t")]).explode("t").drop_nulls()
    sh = l.join(r, on=["pair_i", "t"], how="inner")
    print(f"{tag}: shared rows={sh.height}  distinct tokens={sh['t'].n_unique()}")
    return sh

sh_n = shared(feat, "n1", "n2", "name")
sh_a = shared(feat, "a1", "a2", "addr")
sh_m = shared(feat, "m1", "m2", "num")
UNI = {"name": sh_n["t"].unique(), "addr": sh_a["t"].unique(), "num": sh_m["t"].unique()}
print("universe sizes:", {k: v.len() for k, v in UNI.items()})

del pool, s1samp, gt2, raw, mlist, w, idx, sidx, candx, s1x
gc.collect()
# corpus DFs (universe only):

SRC = [(s1, k1), (s2, k2), (s3, k3)]

def corpus_df(which, pat, uni):
    parts = []
    for i, (src, k) in enumerate(SRC):
        col = k[which]
        v = src.select(N(pl.col(col)).alias("v")).unique()
        p = (v.select(pl.col("v").str.extract_all(pat).list.unique().alias("t"))
               .explode("t")
               .filter(pl.col("t").is_in(uni))
               .group_by("t").len(name="df"))
        print(f"  src{i+1} {which}: distinct texts={v.height}  df rows={p.height}")
        parts.append(p); del v, p; gc.collect()
    return pl.concat(parts).group_by("t").agg(pl.col("df").sum())

df_name = corpus_df("name", r"\w+",    UNI["name"])
df_addr = corpus_df("addr", r"\w+",    UNI["addr"])
df_num  = corpus_df("addr", r"\d{2,}", UNI["num"])
for k in ("name", "addr", "num"):
    got = {"name": df_name, "addr": df_addr, "num": df_num}[k]
    print(f"{k}: universe={UNI[k].len()} covered={got.height} missing={UNI[k].len() - got.height} (expect 0)")
# caps, unions, splits, verdict:

def min_df_join(sh, dff, alias):
    j = sh.join(dff, on="t", how="left")
    miss = int(j["df"].null_count())
    m = j.group_by("pair_i").agg(pl.col("df").min().alias(alias))
    return m, miss

mn, miss_n = min_df_join(sh_n, df_name, "mn")
ma, miss_a = min_df_join(sh_a, df_addr, "ma")
mm, miss_m = min_df_join(sh_m, df_num,  "mm")
print("tokens missing from corpus DF (expect 0):", miss_n, miss_a, miss_m)

f2 = (feat.join(mn, on="pair_i", how="left")
          .join(ma, on="pair_i", how="left")
          .join(mm, on="pair_i", how="left"))
f2 = (f2
    .with_columns([
        (pl.col("mn") <= 300).fill_null(False).alias("n300"),
        (pl.col("ma") <= 300).fill_null(False).alias("a300"),
        (pl.col("mm") <= 100).fill_null(False).alias("m100"),
        (pl.col("mn") <= 1000).fill_null(False).alias("n1000"),
        (pl.col("ma") <= 1000).fill_null(False).alias("a1000"),
        (pl.col("mm") <= 1000).fill_null(False).alias("m1000"),
        pl.col("mn").is_not_null().alias("n_any"),
        pl.col("ma").is_not_null().alias("a_any"),
        pl.col("mm").is_not_null().alias("m_any"),
    ])
    .with_columns([
        (pl.col("n300") | pl.col("a300") | pl.col("m100") | pl.col("exact_fold")).alias("U_strict"),
        (pl.col("n1000") | pl.col("a1000") | pl.col("m1000") | pl.col("exact_fold")).alias("U_loose"),
        (pl.col("n_any") | pl.col("a_any") | pl.col("m_any") | pl.col("exact_fold")).alias("U_uncapped"),
    ]))

COMP = {
    "shared name tok":      "n_any",
    "shared addr tok":      "a_any",
    "shared num >=2d":      "m_any",
    "name tok DF<=300":     "n300",
    "addr tok DF<=300":     "a300",
    "num DF<=100":          "m100",
    "name tok DF<=1000":    "n1000",
    "addr tok DF<=1000":    "a1000",
    "num DF<=1000":         "m1000",
    "exact (norm2)":        "exact",
    "exact (accent-folded)": "exact_fold",
    "U_uncapped":           "U_uncapped",
    "U_strict 300/300/100": "U_strict",
    "U_loose 1000/1000/1000": "U_loose",
}
pos_f = f2.filter(pl.col("y") == 1)
neg_f = f2.filter(pl.col("y") == 0)
print("\n=== component rates (sample-based) ===")
print(f"{'component':26s} {'pos':>8s} {'neg':>8s}")
for label, c in COMP.items():
    print(f"{label:26s} {pos_f[c].mean():8.4f} {neg_f[c].mean():8.4f}")

print("\n=== unions by country ===")
g = (f2.group_by(["s1_country", "y"])
       .agg([pl.col(c).mean().alias(c) for c in ("U_strict","U_loose","U_uncapped")])
       .sort("s1_country", "y"))
for r in g.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  {str(r['s1_country']):12s} {tag}  strict={r['U_strict']:.4f}  loose={r['U_loose']:.4f}  uncapped={r['U_uncapped']:.4f}")

print("\n=== cross-script split ===")
gs = f2.group_by(["cross", "y"]).agg(
    [pl.col(c).mean().alias(c) for c in ("U_strict","U_loose","U_uncapped")])
for r in gs.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  cross={r['cross']} {tag}  strict={r['U_strict']:.4f}  loose={r['U_loose']:.4f}  uncapped={r['U_uncapped']:.4f}")

print("\n=== GO/NO-GO ===")
print(f"U_strict pos={pos_f['U_strict'].mean():.4f}  neg={neg_f['U_strict'].mean():.4f}")
print(f"U_loose  pos={pos_f['U_loose'].mean():.4f}  neg={neg_f['U_loose'].mean():.4f}")
print("GO = pos >= 0.95")

# --- cell 14 ---
import gc, unicodedata
import numpy as np
import polars as pl
from pathlib import Path

# paths
TRAIN = Path("/home/ubuntu/dataset/student_resource/dataset/train")
def find(name):
    p = TRAIN / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def pick(cols, *c):
    for x in c:
        if x in cols: return x
    raise KeyError(f"none of {c} in {list(cols)}")

def cols(df):
    return dict(id=pick(df.columns,"entity_id","id"),
                name=pick(df.columns,"business_name","name"),
                addr=pick(df.columns,"business_address","address"),
                country=pick(df.columns,"country"))

# norm2: uppercase -> strip ASCII punct -> collapse ws
def N(e):
    return (e.str.to_uppercase()
             .str.replace_all(r"[!-/:-@\[-`{-~]", "")
             .str.replace_all(r"\s+", " ")
             .str.strip_chars())

def fold(s):   # accent-fold for the extra exact key
    if s is None: return None
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(ch))

def load(name):
    df = pl.read_csv(find(name), separator="\t")
    i = pick(df.columns, "id", "entity_id", "source_id", "source1_entity_id")
    return df.with_columns(pl.col(i).cast(pl.Utf8))

s1 = load("train_source1.tsv"); k1 = cols(s1)
s2 = load("train_source2.tsv"); k2 = cols(s2)
s3 = load("train_source3.tsv"); k3 = cols(s3)
gt = load("train_ground_truth.tsv")
GID = pick(gt.columns, "source1_entity_id", "s1_id", "id")
GM  = pick(gt.columns, "matched_entity_ids", "matches", "matched_ids")
print("shapes:", s1.shape, s2.shape, s3.shape, gt.shape)

# GT: derive n_matches locally, auto-detect separator
raw = gt[GM].fill_null("")
seps = [",", ";", "|", " "]
sep = max(seps, key=lambda s: raw.str.contains(s, literal=True).fill_null(False).sum())
print("GT separator:", repr(sep))
gt2 = (gt.with_columns(raw.str.split(sep).alias("m"))
         .with_columns(pl.col("m").list.len().alias("n_matches")))
print("mean n_matches (expect ~3.51 incl. empties as 1):", round(gt2["n_matches"].mean(), 4),
      "| singles:", int((gt2["n_matches"] == 1).sum()))

# pos: 25k GT rows weighted by n_matches, exploded
rng = np.random.default_rng(0)
w = gt2["n_matches"].to_numpy().astype("float64"); w /= w.sum()
idx = rng.choice(gt2.height, size=25_000, replace=False, p=w).tolist()
pos = (gt2.with_row_index("ri").filter(pl.col("ri").is_in(idx))
          .explode("m")
          .select([pl.col(GID).alias("s1_id"), pl.col("m").alias("cand_id"),
                   pl.lit(1).alias("y")])
          .with_columns(pl.col("cand_id").str.strip_chars().alias("cand_id"))
          .filter((pl.col("cand_id") != "") & pl.col("cand_id").is_not_null()))

# neg: 40k same-country random pairs
pool = pl.concat([
    s2.select([pl.col(k2["id"]).alias("pid"), pl.col(k2["country"]).alias("ct")]),
    s3.select([pl.col(k3["id"]).alias("pid"), pl.col(k3["country"]).alias("ct")]),
])
sidx = rng.choice(s1.height, size=40_000, replace=False).tolist()
s1samp = s1.with_row_index("ri").filter(pl.col("ri").is_in(sidx))
neg_frames, skipped = [], 0
for i, (key, sub) in enumerate(s1samp.group_by(k1["country"])):
    ct = key[0]
    cand = pool.filter(pl.col("ct") == ct)
    if cand.height == 0:
        skipped += len(sub); continue
    take = cand.sample(n=len(sub), with_replacement=True, seed=100 + i)
    neg_frames.append(pl.DataFrame({
        "s1_id": sub[k1["id"]].to_numpy(),
        "cand_id": take["pid"].to_numpy(),
        "y": np.zeros(len(sub), dtype=np.int32)}))
neg = pl.concat(neg_frames)
print("neg pairs:", neg.height, "| skipped:", skipped)

pairs = pl.concat([pos, neg]).with_row_index("pair_i")
print("pairs:", pairs.height, "| pos:", int((pairs["y"]==1).sum()),
      "| neg:", int((pairs["y"]==0).sum()))

# text pull via isin (derived vars only)
all_s1 = pairs["s1_id"].unique().to_list()
all_c  = pairs["cand_id"].unique().to_list()
s1x = s1.filter(pl.col(k1["id"]).is_in(all_s1)).select([
    pl.col(k1["id"]).alias("s1_id"), pl.col(k1["name"]).alias("s1_name"),
    pl.col(k1["addr"]).alias("s1_addr"), pl.col(k1["country"]).alias("s1_country")])
candx = pl.concat([
    s2.filter(pl.col(k2["id"]).is_in(all_c)).select([
        pl.col(k2["id"]).alias("cid"), pl.col(k2["name"]).alias("c_name"),
        pl.col(k2["addr"]).alias("c_addr"), pl.col(k2["country"]).alias("c_country")]),
    s3.filter(pl.col(k3["id"]).is_in(all_c)).select([
        pl.col(k3["id"]).alias("cid"), pl.col(k3["name"]).alias("c_name"),
        pl.col(k3["addr"]).alias("c_addr"), pl.col(k3["country"]).alias("c_country")]),
])
feat = (pairs.join(s1x, on="s1_id", how="left")
              .join(candx, left_on="cand_id", right_on="cid", how="left"))
print("pull nulls (expect 0,0):", feat["s1_name"].null_count(), feat["c_name"].null_count())
print("countries:", {r[0]: r[1] for r in feat.group_by("s1_country").len().rows()})

# per-pair keys
DEV = r"[\u0900-\u097F]"
feat = (feat
    .with_columns([
        N(pl.col("s1_name")).alias("sn1"), N(pl.col("c_name")).alias("sn2"),
        N(pl.col("s1_addr")).alias("sa1"), N(pl.col("c_addr")).alias("sa2"),
        pl.col("s1_name").str.contains(DEV).fill_null(False).alias("d1"),
        pl.col("c_name").str.contains(DEV).fill_null(False).alias("d2"),
    ])
    .with_columns([
        pl.col("sn1").str.extract_all(r"\w+").list.unique().alias("n1"),
        pl.col("sn2").str.extract_all(r"\w+").list.unique().alias("n2"),
        pl.col("sa1").str.extract_all(r"\w+").list.unique().alias("a1"),
        pl.col("sa2").str.extract_all(r"\w+").list.unique().alias("a2"),
        pl.col("sa1").str.extract_all(r"\d{2,}").list.unique().alias("m1"),
        pl.col("sa2").str.extract_all(r"\d{2,}").list.unique().alias("m2"),
        ((pl.col("sn1") == pl.col("sn2")) & (pl.col("sn1").str.len_chars() > 0)).alias("exact"),
        pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("f1"),
        pl.col("sn2").map_elements(fold, return_dtype=pl.Utf8).alias("f2"),
    ])
    .with_columns([
        ((pl.col("f1") == pl.col("f2")) & (pl.col("f1").str.len_chars() > 0)).fill_null(False).alias("exact_fold"),
        (pl.col("d1") != pl.col("d2")).alias("cross"),
    ]))

def shared(df, c1, c2, tag):
    l = df.select(["pair_i", pl.col(c1).alias("t")]).explode("t").drop_nulls()
    r = df.select(["pair_i", pl.col(c2).alias("t")]).explode("t").drop_nulls()
    sh = l.join(r, on=["pair_i", "t"], how="inner")
    print(f"{tag}: shared rows={sh.height}  distinct tokens={sh['t'].n_unique()}")
    return sh

sh_n = shared(feat, "n1", "n2", "name")
sh_a = shared(feat, "a1", "a2", "addr")
sh_m = shared(feat, "m1", "m2", "num")
UNI = {"name": sh_n["t"].unique(), "addr": sh_a["t"].unique(), "num": sh_m["t"].unique()}
print("universe sizes:", {k: v.len() for k, v in UNI.items()})

del pool, s1samp, gt2, raw, w, idx, sidx, candx, s1x, neg_frames
gc.collect()
# record-level corpus DFs:

SRC = [(s1, k1), (s2, k2), (s3, k3)]

def corpus_df(which, pat, uni):
    uni_l = uni.to_list()
    parts = []
    for i, (src, k) in enumerate(SRC):
        p = (src.select(N(pl.col(k[which])).str.extract_all(pat).list.unique().alias("t"))
               .explode("t")
               .filter(pl.col("t").is_in(uni_l))
               .group_by("t").len(name="df"))
        print(f"  src{i+1} {which}: df rows={p.height}")
        parts.append(p); gc.collect()
    return pl.concat(parts).group_by("t").agg(pl.col("df").sum())

df_name = corpus_df("name", r"\w+",    UNI["name"])
df_addr = corpus_df("addr", r"\w+",    UNI["addr"])
df_num  = corpus_df("addr", r"\d{2,}", UNI["num"])
for kk, d in (("name", df_name), ("addr", df_addr), ("num", df_num)):
    print(f"{kk}: universe={UNI[kk].len()} covered={d.height} missing={UNI[kk].len()-d.height} (expect 0)")
# min-DF + unions (the cells that must re-run against the new DFs):

def min_df_join(sh, dff, alias):
    j = sh.join(dff, on="t", how="left")
    miss = int(j["df"].null_count())
    m = j.group_by("pair_i").agg(pl.col("df").min().alias(alias))
    return m, miss

mn, miss_n = min_df_join(sh_n, df_name, "mn")
ma, miss_a = min_df_join(sh_a, df_addr, "ma")
mm, miss_m = min_df_join(sh_m, df_num,  "mm")
print("tokens missing from corpus DF (expect 0):", miss_n, miss_a, miss_m)

f2 = (feat.join(mn, on="pair_i", how="left")
          .join(ma, on="pair_i", how="left")
          .join(mm, on="pair_i", how="left"))
f2 = (f2
    .with_columns([
        (pl.col("mn") <= 300).fill_null(False).alias("n300"),
        (pl.col("ma") <= 300).fill_null(False).alias("a300"),
        (pl.col("mm") <= 100).fill_null(False).alias("m100"),
        (pl.col("mn") <= 1000).fill_null(False).alias("n1000"),
        (pl.col("ma") <= 1000).fill_null(False).alias("a1000"),
        (pl.col("mm") <= 1000).fill_null(False).alias("m1000"),
        pl.col("mn").is_not_null().alias("n_any"),
        pl.col("ma").is_not_null().alias("a_any"),
        pl.col("mm").is_not_null().alias("m_any"),
    ])
    .with_columns([
        (pl.col("n300") | pl.col("a300") | pl.col("m100") | pl.col("exact_fold")).alias("U_strict"),
        (pl.col("n1000") | pl.col("a1000") | pl.col("m1000") | pl.col("exact_fold")).alias("U_loose"),
        (pl.col("n_any") | pl.col("a_any") | pl.col("m_any") | pl.col("exact_fold")).alias("U_uncapped"),
    ]))

COMP = {
    "shared name tok":      "n_any",
    "shared addr tok":      "a_any",
    "shared num >=2d":      "m_any",
    "name tok DF<=300":     "n300",
    "addr tok DF<=300":     "a300",
    "num DF<=100":          "m100",
    "name tok DF<=1000":    "n1000",
    "addr tok DF<=1000":    "a1000",
    "num DF<=1000":         "m1000",
    "exact (norm2)":        "exact",
    "exact (accent-folded)": "exact_fold",
    "U_uncapped":           "U_uncapped",
    "U_strict 300/300/100": "U_strict",
    "U_loose 1000/1000/1000": "U_loose",
}
pos_f = f2.filter(pl.col("y") == 1)
neg_f = f2.filter(pl.col("y") == 0)
print("\n=== component rates (record-level DF) ===")
print(f"{'component':26s} {'pos':>8s} {'neg':>8s}")
for label, c in COMP.items():
    print(f"{label:26s} {pos_f[c].mean():8.4f} {neg_f[c].mean():8.4f}")

print("\n=== unions by country ===")
g = (f2.group_by(["s1_country", "y"])
       .agg([pl.col(c).mean().alias(c) for c in ("U_strict","U_loose","U_uncapped")])
       .sort("s1_country", "y"))
for r in g.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  {str(r['s1_country']):12s} {tag}  strict={r['U_strict']:.4f}  loose={r['U_loose']:.4f}  uncapped={r['U_uncapped']:.4f}")

print("\n=== cross-script split ===")
gs = f2.group_by(["cross", "y"]).agg(
    [pl.col(c).mean().alias(c) for c in ("U_strict","U_loose","U_uncapped")])
for r in gs.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  cross={r['cross']} {tag}  strict={r['U_strict']:.4f}  loose={r['U_loose']:.4f}  uncapped={r['U_uncapped']:.4f}")

print("\n=== EDA-10 corrected unions ===")
print(f"U_strict pos={pos_f['U_strict'].mean():.4f}  neg={neg_f['U_strict'].mean():.4f}")
print(f"U_loose  pos={pos_f['U_loose'].mean():.4f}  neg={neg_f['U_loose'].mean():.4f}")
print(f"U_uncapped pos={pos_f['U_uncapped'].mean():.4f}  neg={neg_f['U_uncapped'].mean():.4f}")
# EDA-11 cap sweep:

# EDA-11: cap sweep — recall / neg / sum-UB candidates-per-S1 across DF caps
CAPS = [100, 300, 1000, 3000, 10000, 30000, 100000, 10**12]
COMBOS = sorted({(c, c) for c in CAPS} | {(300, 100), (1000, 1000)})

def uexpr(ca, cn):
    return ((pl.col("mn") <= ca).fill_null(False)
            | (pl.col("ma") <= ca).fill_null(False)
            | (pl.col("mm") <= cn).fill_null(False)
            | pl.col("exact_fold"))

# per-S1 token cost (sum-UB, pre-dedup), built ONCE
s1tok = f2.select(["s1_id", "sn1", "n1", "a1", "m1"]).unique(subset=["s1_id"])
def explode_join(listcol, dff):
    e = s1tok.select(["s1_id", pl.col(listcol).alias("t")]).explode("t").drop_nulls()
    return e.join(dff, on="t", how="inner")   # (s1_id, t, df)
en = explode_join("n1", df_name)
ea = explode_join("a1", df_addr)
em = explode_join("m1", df_num)

# exact-key cost (pool s2+s3, record-level, no unique)
keys = s1tok["sn1"].unique().to_list()
eparts = []
for src, k in [(s2, k2), (s3, k3)]:
    v = src.select(N(pl.col(k["name"])).alias("k")).filter(pl.col("k").is_in(keys))
    eparts.append(v.group_by("k").len(name="df"))
edf = pl.concat(eparts).group_by("k").agg(pl.col("df").sum())
exact_per_s1 = s1tok.join(edf, left_on="sn1", right_on="k", how="left")["df"].fill_null(0)
mean_exact = exact_per_s1.mean()
print(f"exact-key cost (pool): mean {mean_exact:.2f}/S1  (expect ~9.3)\n")

def cost(capA, capN):
    parts = []
    for i, (en_, cap) in enumerate([(en, capA), (ea, capA), (em, capN)]):
        parts.append(en_.filter(pl.col("df") <= cap)
                         .group_by("s1_id").agg(pl.col("df").sum().alias(f"c{i}")))
    tot = s1tok.select("s1_id")
    for g in parts:
        tot = tot.join(g, on="s1_id", how="left")
    tot = tot.with_columns([pl.col(f"c{i}").fill_null(0) for i in range(3)])
    s = tot.with_columns((pl.col("c0") + pl.col("c1") + pl.col("c2")).alias("cand"))["cand"]
    return s.mean(), s.quantile(0.99), float((s == 0).mean()) * 100

print(f"{'capA':>7} {'capN':>7} | {'posRecall':>9} {'negRate':>8} | {'sumUB mean':>10} {'p99':>7} {'zero%':>6} | {'total+exact':>11}")
for ca, cn in COMBOS:
    p = pos_f.select(uexpr(ca, cn).alias("r"))["r"].mean()
    n = neg_f.select(uexpr(ca, cn).alias("r"))["r"].mean()
    m, q, z = cost(ca, cn)
    tag = " <= continuity w/ Cell 3" if (ca, cn) in [(300, 100), (1000, 1000)] else (" <- uncapped" if ca > 10**11 else "")
    print(f"{ca:>7} {cn:>7} | {p:9.4f} {n:8.4f} | {m:10.1f} {q:7.0f} {z:6.1f} | {m + mean_exact:11.1f}{tag}")

# --- cell 15 ---
# EDA-12: pair-key (2-token / cross-field) recall ceilings from in-memory sh_*
cntN = sh_n.group_by("pair_i").len().rename({"len": "cntN"})
cntA = sh_a.group_by("pair_i").len().rename({"len": "cntA"})
cntM = sh_m.group_by("pair_i").len().rename({"len": "cntM"})
g = (f2.join(cntN, on="pair_i", how="left")
        .join(cntA, on="pair_i", how="left")
        .join(cntM, on="pair_i", how="left")
        .with_columns([pl.col("cntN").fill_null(0),
                       pl.col("cntA").fill_null(0),
                       pl.col("cntM").fill_null(0)]))
posf = g.filter(pl.col("y") == 1)
negf = g.filter(pl.col("y") == 0)

print("=== shared-token counts on POS pairs ===")
for c in ("cntN", "cntA", "cntM"):
    s = posf[c]
    qs = {q: s.quantile(q) for q in (0.25, 0.5, 0.75, 0.9)}
    print(f"{c}: mean={s.mean():.2f}  zeros={float((s==0).mean())*100:.1f}%  q={qs}")

f_n2 = pl.col("cntN") >= 2
f_a2 = pl.col("cntA") >= 2
f_x  = (pl.col("cntN") >= 1) & (pl.col("cntA") >= 1)
f_xn = (pl.col("cntN") >= 1) & (pl.col("cntM") >= 1)
f_xa = (pl.col("cntA") >= 1) & (pl.col("cntM") >= 1)
f_m2 = pl.col("cntM") >= 2
FAM = {"N>=2 name-pair": f_n2, "A>=2 addr-pair": f_a2, "cross N&A": f_x,
       "cross N&M": f_xn, "cross A&M": f_xa, "M>=2": f_m2}

print("\n=== family rates (pos | neg) ===")
for k, e in FAM.items():
    print(f"{k:18s} {posf.select(e.alias('r'))['r'].mean():8.4f} {negf.select(e.alias('r'))['r'].mean():8.4f}")

u_pair  = f_n2 | f_a2 | f_x | f_xn | f_xa | f_m2
u_p_e   = u_pair | pl.col("exact_fold")
u_p_s_e = u_pair | pl.col("exact_fold") | pl.col("n300") | pl.col("a300") | pl.col("m100")

print("\n=== pair-key unions ===")
for name, e in {"U_pair": u_pair, "U_pair+exact": u_p_e,
                "U_pair+exact+tokens300": u_p_s_e}.items():
    print(f"{name:24s} pos={posf.select(e.alias('r'))['r'].mean():.4f}  neg={negf.select(e.alias('r'))['r'].mean():.4f}")

g2 = g.with_columns([u_p_e.alias("u1"), u_p_s_e.alias("u2")])
print("\n=== U_pair+exact / +tokens300 by country ===")
gg = g2.group_by(["s1_country", "y"]).agg([pl.col("u1").mean(), pl.col("u2").mean()]).sort("s1_country", "y")
for r in gg.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  {str(r['s1_country']):10s} {tag}  pair+exact={r['u1']:.4f}  +tokens={r['u2']:.4f}")
print("=== cross-script ===")
gc_ = g2.group_by(["cross", "y"]).agg([pl.col("u1").mean(), pl.col("u2").mean()])
for r in gc_.iter_rows(named=True):
    tag = "pos" if r["y"] == 1 else "neg"
    print(f"  cross={r['cross']} {tag}  pair+exact={r['u1']:.4f}  +tokens={r['u2']:.4f}")

# --- cell 16 ---
# EDA-13a: pool postings (s2+s3) — global pid = row_index (+ s2.height offset for s3)
s1tok = f2.select(["s1_id", "sn1", "n1", "a1", "m1"]).unique(subset=["s1_id"])
tokN = s1tok.select(pl.col("n1").explode().alias("t")).drop_nulls().unique()
tokA = s1tok.select(pl.col("a1").explode().alias("t")).drop_nulls().unique()
tokM = s1tok.select(pl.col("m1").explode().alias("t")).drop_nulls().unique()
print("universe tokens: name", tokN.height, "addr", tokA.height, "num", tokM.height)

N2 = s2.height  # offset for s3 pids

def build_post(which, pat, tok_df):
    tok_l = tok_df["t"].to_list()
    parts = []
    for off, (src, k) in enumerate([(s2, k2), (s3, k3)]):
        p = (src.with_row_index("pid")
               .select(["pid", N(pl.col(k[which])).str.extract_all(pat).list.unique().alias("t")])
               .explode("t")
               .filter(pl.col("t").is_in(tok_l))
               .select(["t", pl.col("pid").alias("rid")]))
        if off == 1:
            p = p.with_columns((pl.col("rid") + N2).alias("rid"))
        parts.append(p); gc.collect()
    post = pl.concat(parts)
    tmap = post.select("t").unique().with_row_index("tid")
    post = post.join(tmap, on="t").select(["tid", "rid"]).sort(["tid", "rid"])
    return post, dict(zip(tmap["t"].to_list(), tmap["tid"].to_list()))

postN, mapN = build_post("name", r"\w+",    tokN)
postA, mapA = build_post("addr", r"\w+",    tokA)
postM, mapM = build_post("addr", r"\d{2,}", tokM)
print("postings rows:", postN.height, postA.height, postM.height)

def to_dict(post):
    d = {}
    for tid, rids in post.group_by("tid").agg(pl.col("rid")).iter_rows():
        d[int(tid)] = np.asarray(rids, dtype=np.int64)   # sorted: frame sorted by (tid, rid)
    return d
P_N, P_A, P_M = to_dict(postN), to_dict(postA), to_dict(postM)
del postN, postA, postM, tokN, tokA, tokM
gc.collect()

# exact-fold buckets: fold over pool names, keep only sample keys
fold_keys = s1tok.select(pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk")).unique()
fk_l = fold_keys["fk"].to_list()
eparts = []
for off, (src, k) in enumerate([(s2, k2), (s3, k3)]):
    v = (src.with_row_index("pid")
           .select(["pid", N(pl.col(k["name"])).map_elements(fold, return_dtype=pl.Utf8).alias("fk")])
           .filter(pl.col("fk").is_in(fk_l))
           .select(["fk", pl.col("pid").alias("rid")]))
    if off == 1:
        v = v.with_columns((pl.col("rid") + N2).alias("rid"))
    eparts.append(v); gc.collect()
exact_post = pl.concat(eparts).sort(["fk", "rid"])
EX = {fk: np.asarray(r, dtype=np.int64) for fk, r in exact_post.group_by("fk").agg(pl.col("rid")).iter_rows()}
print("exact-fold postings rows:", exact_post.height, "| buckets:", len(EX))
del eparts, exact_post, fold_keys, fk_l
gc.collect()

# EDA-13b: greedy key selection at budget -> recall & exact distinct candidates
MAXB = 200
BUDGETS = [10, 25, 50, 100, 200]
PMAP = {"N": P_N, "A": P_A, "M": P_M}

# S1 (str) -> true matches (global pid int) ; S1 -> country
pp = f2.filter(pl.col("y") == 1).select(["s1_id", "cand_id"]).unique()
cand_l = pp["cand_id"].unique().to_list()
t_parts = []
for off, (src, k) in enumerate([(s2, k2), (s3, k3)]):
    t = (src.with_row_index("pid")
           .select([pl.col(k["id"]).alias("eid"), "pid"])
           .filter(pl.col("eid").is_in(cand_l)))
    if off == 1:
        t = t.with_columns((pl.col("pid") + N2).alias("pid"))
    t_parts.append(t.select(pl.col("eid").alias("cand_id"), pl.col("pid").alias("gpid")))
tr = pl.concat(t_parts)
pp2 = pp.join(tr, on="cand_id", how="left")
print("match ids NOT found in pool (expect 0):", pp2["gpid"].null_count())
match_map = {}
for s, _, g in pp2.iter_rows():
    match_map.setdefault(s, set()).add(int(g))
ct_map = {r["s1_id"]: r["s1_country"]
          for r in f2.select(["s1_id", "s1_country"]).unique(subset=["s1_id"]).iter_rows(named=True)}
print("pos pairs in match_map:", sum(len(v) for v in match_map.values()))

def get_b(kind, t):
    a = PMAP[kind].get(t)
    return None if a is None or len(a) == 0 else a

def inter_count(a, b):
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a)
    ok = idx < len(b)
    return int(np.count_nonzero(ok & (b[np.minimum(idx, len(b) - 1)] == a)))

def eval_pair(k1, t1, k2, t2):
    a, b = get_b(k1, t1), get_b(k2, t2)
    if a is None or b is None: return None
    df = inter_count(a, b)
    if df == 0 or df > MAXB: return None
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a); ok = idx < len(b)
    hits = a[ok & (b[np.minimum(idx, len(b) - 1)] == a)]
    return (df, hits) if hits.size else None

def build_keys(row):
    n_t = [mapN[t] for t in (row["n1"] or []) if t in mapN]
    a_t = [mapA[t] for t in (row["a1"] or []) if t in mapA]
    m_t = [mapM[t] for t in (row["m1"] or []) if t in mapM]
    ev = []   # (df, arr, solo)
    for kind, toks in (("N", n_t), ("A", a_t), ("M", m_t)):
        for t in toks:
            arr = get_b(kind, t)
            if arr is not None and len(arr) <= MAXB:
                ev.append((len(arr), arr, True))
    ex = EX.get(row["fk"])
    if ex is not None and 0 < len(ex) <= MAXB:
        ev.append((len(ex), ex, True))
    for arr, kind in ((n_t, "N"), (a_t, "A"), (m_t, "M")):
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                r = eval_pair(kind, arr[i], kind, arr[j])
                if r: ev.append((r[0], r[1], False))
    for x, kx in ((n_t, "N"), (a_t, "A")):
        for y, ky in ((a_t, "A"), (m_t, "M")):
            if kx == ky: continue
            for t1 in x:
                for t2 in y:
                    r = eval_pair(kx, t1, ky, t2)
                    if r: ev.append((r[0], r[1], False))
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

res = {b: {"cov": 0, "tot": 0, "cands": [], "ct_cov": {}, "ct_tot": {}} for b in BUDGETS}
abl = {"cov": 0, "tot": 0, "cands": []}
n_keys = 0
s1rows = s1tok.with_columns(pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk"))
for row in s1rows.iter_rows(named=True):
    s_id = row["s1_id"]
    ev = build_keys(row)
    n_keys += len(ev)
    matches = match_map.get(s_id, set())
    ct = ct_map.get(s_id, "?")
    for b in BUDGETS:
        u = run_greedy(ev, b)
        res[b]["cands"].append(len(u))
        cov = sum(1 for m in matches if m in u)
        res[b]["cov"] += cov; res[b]["tot"] += len(matches)
        res[b]["ct_cov"][ct] = res[b]["ct_cov"].get(ct, 0) + cov
        res[b]["ct_tot"][ct] = res[b]["ct_tot"].get(ct, 0) + len(matches)
    u = run_greedy(ev, 50, solo_only=True)
    abl["cands"].append(len(u))
    abl["cov"] += sum(1 for m in matches if m in u); abl["tot"] += len(matches)

print(f"mean evaluated keys/S1: {n_keys / s1rows.height:.1f}\n")
print(f"{'budget':>7} | {'recall':>7} | {'mean':>7} {'med':>5} {'p99':>5} {'max':>5} {'zero%':>6}")
for b in BUDGETS:
    r = res[b]; cs = np.array(r["cands"])
    print(f"{b:>7} | {r['cov']/max(r['tot'],1):7.4f} | {cs.mean():7.1f} {np.median(cs):5.0f} "
          f"{np.percentile(cs,99):5.0f} {cs.max():5.0f} {float((cs==0).mean())*100:6.1f}")
print(f"\nsingles+exact ONLY @50: recall={abl['cov']/max(abl['tot'],1):.4f}  mean={np.mean(abl['cands']):.1f}")
print("recall @50 by country:", {k: round(v/max(res[50]['ct_tot'][k],1), 4)
      for k, v in res[50]["ct_cov"].items()})

# --- cell 17 ---
import pickle
with open("block_state.pkl", "wb") as f:
    pickle.dump(dict(P_N=P_N, P_A=P_A, P_M=P_M, mapN=mapN, mapA=mapA, mapM=mapM, EX=EX,
                     match_map=match_map, ct_map=ct_map, s1tok=s1tok, f2=f2), f)
print("state saved")

# --- cell 18 ---
# EDA-14: miss autopsy at budget 50 (pos S1s only) -> capacity vs big-key vs no-shared
stat = {"covered": 0, "capacity": 0, "big_single": 0, "exact_big": 0, "anomaly": 0, "no_shared": 0}
by_ct = {}
pos_rows = s1rows.filter(pl.col("s1_id").is_in(list(match_map.keys())))
print("pos S1 rows:", pos_rows.height)
n_miss_tot = 0

for row in pos_rows.iter_rows(named=True):
    s_id = row["s1_id"]
    matches = match_map.get(s_id, set())
    if not matches:
        continue
    ev = build_keys(row)
    U50 = run_greedy(ev, 50)
    Uall = set()
    for _, arr, _ in ev:
        Uall.update(arr.tolist())
    fk = row["fk"]
    ct_stat = by_ct.setdefault(ct_map.get(s_id, "?"),
                               dict.fromkeys(stat, 0))
    for m in matches:
        if m in U50:
            stat["covered"] += 1; ct_stat["covered"] += 1; continue
        n_miss_tot += 1
        if m in Uall:
            stat["capacity"] += 1; ct_stat["capacity"] += 1; continue
        # not in any evaluated key -> why?
        found = None
        for kind, key, PM in (("N", "n1", P_N), ("A", "a1", P_A), ("M", "m1", P_M)):
            for tok in (row[key] or []):
                tid = mapN.get(tok) if kind == "N" else mapA.get(tok) if kind == "A" else mapM.get(tok)
                if tid is None:
                    continue
                arr = PM.get(tid)
                if arr is not None and np.isin(np.array([m]), arr)[0]:
                    found = "big_single" if len(arr) > MAXB else "anomaly"
                    break
            if found:
                break
        if found is None:
            ex = EX.get(fk)
            if ex is not None and np.isin(np.array([m]), ex)[0]:
                found = "exact_big" if len(ex) > MAXB else "anomaly"
        found = found or "no_shared"
        stat[found] += 1; ct_stat[found] += 1

tot = stat["covered"] + n_miss_tot
print(f"\ntotal pos pairs: {tot} (expect 105,684) | miss@50: {n_miss_tot}")
for k in ["covered", "capacity", "big_single", "exact_big", "anomaly", "no_shared"]:
    print(f"{k:>10}: {stat[k]:6d}  {stat[k]/tot:7.4f}")
print("\nby country (fractions):")
for ct, d in by_ct.items():
    t = d["covered"] + d["capacity"] + d["big_single"] + d["exact_big"] + d["anomaly"] + d["no_shared"]
    print(f"  {ct:6}: cov {d['covered']/t:.4f} | cap {d['capacity']/t:.4f} | big {d['big_single']/t:.4f} "
          f"| exbig {d['exact_big']/t:.4f} | anom {d['anomaly']/t:.4f} | none {d['no_shared']/t:.4f}")

# --- cell 19 ---
import pickle, unicodedata
import numpy as np
import polars as pl

with open("block_state.pkl", "rb") as f:
    S = pickle.load(f)
P_N, P_A, P_M = S["P_N"], S["P_A"], S["P_M"]
mapN, mapA, mapM = S["mapN"], S["mapA"], S["mapM"]
EX, match_map = S["EX"], S["match_map"]
ct_map, s1tok = S["ct_map"], S["s1tok"]

def fold(s):   # matches training.ipynb fold() - EX keys depend on this
    if s is None: return None
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(ch))

s1rows = s1tok.with_columns(pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk"))
MAXB = 200
PMAP = {"N": P_N, "A": P_A, "M": P_M}

def get_b(kind, t):
    a = PMAP[kind].get(t)
    return None if a is None or len(a) == 0 else a

def inter_count(a, b):
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a)
    ok = idx < len(b)
    return int(np.count_nonzero(ok & (b[np.minimum(idx, len(b) - 1)] == a)))

def eval_pair(k1, t1, k2, t2):
    a, b = get_b(k1, t1), get_b(k2, t2)
    if a is None or b is None: return None
    df = inter_count(a, b)
    if df == 0 or df > MAXB: return None
    if len(a) > len(b): a, b = b, a
    idx = np.searchsorted(b, a); ok = idx < len(b)
    hits = a[ok & (b[np.minimum(idx, len(b) - 1)] == a)]
    return (df, hits) if hits.size else None

def build_keys(row):
    n_t = [mapN[t] for t in (row["n1"] or []) if t in mapN]
    a_t = [mapA[t] for t in (row["a1"] or []) if t in mapA]
    m_t = [mapM[t] for t in (row["m1"] or []) if t in mapM]
    ev = []
    for kind, toks in (("N", n_t), ("A", a_t), ("M", m_t)):
        for t in toks:
            arr = get_b(kind, t)
            if arr is not None and len(arr) <= MAXB:
                ev.append((len(arr), arr, True))
    ex = EX.get(row["fk"])
    if ex is not None and 0 < len(ex) <= MAXB:
        ev.append((len(ex), ex, True))
    for arr, kind in ((n_t, "N"), (a_t, "A"), (m_t, "M")):
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                r = eval_pair(kind, arr[i], kind, arr[j])
                if r: ev.append((r[0], r[1], False))
    for x, kx in ((n_t, "N"), (a_t, "A")):
        for y, ky in ((a_t, "A"), (m_t, "M")):
            if kx == ky: continue
            for t1 in x:
                for t2 in y:
                    r = eval_pair(kx, t1, ky, t2)
                    if r: ev.append((r[0], r[1], False))
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

print("restored | EX:", len(EX), "| pos pairs:", sum(len(v) for v in match_map.values()), "| s1tok:", s1tok.height)

# --- cell 20 ---
# EDA-15: joint MAXB x budget grid, ev built ONCE per row (MAXB=inf), filter per config
GRID_M = [200, 1000, 5000, 10**9]        # 10**9 = effectively inf
GRID_B = [50, 100, 200]
LABEL  = {200: "200", 1000: "1000", 5000: "5000", 10**9: "inf"}
SAVED_MAXB, MAXB = MAXB, 10**9           # build_keys/eval_pair read global MAXB
grid = {(m, b): {"cov": 0, "tot": 0, "cands": []} for m in GRID_M for b in GRID_B}
pos_rows = s1rows.filter(pl.col("s1_id").is_in(list(match_map.keys())))
n = 0
for row in pos_rows.iter_rows(named=True):
    matches = match_map.get(row["s1_id"], set())
    if not matches:
        continue
    ev = build_keys(row)                 # keys computed once, uncapped
    for m_cfg in GRID_M:
        ev_m = [e for e in ev if e[0] <= m_cfg]
        for b_cfg in GRID_B:
            u = run_greedy(ev_m, b_cfg)
            g = grid[(m_cfg, b_cfg)]
            g["cands"].append(len(u))
            g["cov"] += sum(1 for mm in matches if mm in u)
            g["tot"] += len(matches)
    n += 1
MAXB = SAVED_MAXB
print("pos S1 rows:", n, "(expect 24601)\n")
print(f"{'MAXB':>5} | {'budget':>6} | {'recall':>7} | {'mean':>6} {'med':>5} {'p99':>5} {'zero%':>6}")
for m_cfg in GRID_M:
    for b_cfg in GRID_B:
        g = grid[(m_cfg, b_cfg)]
        cs = np.array(g["cands"])
        print(f"{LABEL[m_cfg]:>5} | {b_cfg:6} | {g['cov']/max(g['tot'],1):7.4f} | "
              f"{cs.mean():6.1f} {np.median(cs):5.0f} {np.percentile(cs,99):5.0f} {float((cs==0).mean())*100:6.1f}")

# --- cell 21 ---
# EDA-16 T1: test load + 20k/country sample + test-pool postings (2 passes/source, string-keyed)
import numpy as np, polars as pl, unicodedata, gc, time
BASE = "/home/ubuntu/dataset/student_resource/dataset"
rng = np.random.default_rng(0)

def N(e):   # matches training.ipynb N()
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
            if p.lower() in c.lower():
                return c
    raise ValueError(f"{pats} not in {cols}")

t1 = pl.read_csv(f"{BASE}/test/test_source1.tsv", separator="\t")
t2 = pl.read_csv(f"{BASE}/test/test_source2.tsv", separator="\t")
t3 = pl.read_csv(f"{BASE}/test/test_source3.tsv", separator="\t")
print("t1/t2/t3:", t1.height, t2.height, t3.height)   # expect 1732544 / 4887273 / 5082316

nc, ac, cc = pick(t1.columns, "business_name", "name"), pick(t1.columns, "business_address", "address"), pick(t1.columns, "country")
nc2, ac2 = pick(t2.columns, "business_name", "name"), pick(t2.columns, "business_address", "address")
nc3, ac3 = pick(t3.columns, "business_name", "name"), pick(t3.columns, "business_address", "address")
print("cols:", nc, "|", ac, "|", cc)

# stratified sample: 20k per country, seed 0
t1 = t1.with_columns(pl.Series("rand", rng.random(t1.height)))
samp = t1.sort("rand").group_by(cc).head(20000).drop("rand")
fc = {r[0]: r[1] for r in t1[cc].value_counts().iter_rows()}   # FULL country counts for projection
print("sample:", samp.height, "| full counts:", fc)

# S1 features
samp = (samp.with_columns(N(pl.col(nc)).alias("sn1"))
        .with_columns([pl.col("sn1").str.extract_all(r"\w+").list.unique().alias("n1"),
                       N(pl.col(ac)).alias("na"),
                       pl.col("sn1").map_elements(fold, return_dtype=pl.Utf8).alias("fk")])
        .with_columns([pl.col("na").str.extract_all(r"\w+").list.unique().alias("a1"),
                       pl.col("na").str.extract_all(r"\d{2,}").list.unique().alias("m1")])
        .drop("na"))

tokN_l = samp.select(pl.col("n1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokA_l = samp.select(pl.col("a1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
tokM_l = samp.select(pl.col("m1").explode().alias("t")).drop_nulls().unique()["t"].to_list()
fk_list = samp.select("fk").unique()["fk"].to_list()
print("universe:", len(tokN_l), len(tokA_l), len(tokM_l), "| exact keys:", len(fk_list))

# test-pool postings
PT_N, PT_A, PT_M, EXT = {}, {}, {}, {}
NT2 = t2.height
def merge(dst, groups):
    for key, arr in groups:
        dst[key] = np.concatenate([dst[key], arr]) if key in dst else arr

for off, (src, ncx, acx) in enumerate([(t2, nc2, ac2), (t3, nc3, ac3)]):
    # pass 1: names -> exact-fold + name postings
    base = (src.with_row_index("pid")
              .with_columns(N(pl.col(ncx)).alias("nm"))
              .with_columns(pl.col("nm").map_elements(fold, return_dtype=pl.Utf8).alias("fk"))
              .select(["pid", "nm", "fk"]))
    ex = base.filter(pl.col("fk").is_in(fk_list)).select(["fk", pl.col("pid").alias("rid")])
    if off: ex = ex.with_columns((pl.col("rid") + NT2).alias("rid"))
    merge(EXT, ((k, np.asarray(r, dtype=np.int32))
                for k, r in ex.group_by("fk").agg(pl.col("rid").sort()).iter_rows()))
    del ex
    pn = (base.select(["pid", pl.col("nm").str.extract_all(r"\w+").alias("t")])
              .explode("t").filter(pl.col("t").is_in(tokN_l))
              .select(["t", pl.col("pid").alias("rid")]))
    if off: pn = pn.with_columns((pl.col("rid") + NT2).alias("rid"))
    merge(PT_N, ((k, np.asarray(r, dtype=np.int32))
                 for k, r in pn.group_by("t").agg(pl.col("rid").sort()).iter_rows()))
    del base, pn
    gc.collect()
    # pass 2: addr -> words + numbers (single normalization)
    ba = (src.with_row_index("pid").with_columns(N(pl.col(acx)).alias("na")).select(["pid", "na"]))
    for dst, pat, toks in ((PT_A, r"\w+", tokA_l), (PT_M, r"\d{2,}", tokM_l)):
        pa = (ba.select(["pid", pl.col("na").str.extract_all(pat).alias("t")])
                .explode("t").filter(pl.col("t").is_in(toks))
                .select(["t", pl.col("pid").alias("rid")]))
        if off: pa = pa.with_columns((pl.col("rid") + NT2).alias("rid"))
        merge(dst, ((k, np.asarray(r, dtype=np.int32))
                    for k, r in pa.group_by("t").agg(pl.col("rid").sort()).iter_rows()))
        del pa
    del ba
    gc.collect()
    print("pool source done:", off, "| N", len(PT_N), "A", len(PT_A), "M", len(PT_M), "EX", len(EXT))

del t2, t3
gc.collect()
print("posting rows:", sum(len(v) for v in PT_N.values()),
      sum(len(v) for v in PT_A.values()), sum(len(v) for v in PT_M.values()))

# EDA-16 T2: greedy @ {50,100,200} on sampled test S1 - candidate counts by country + projections
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

# run_greedy from the blocking-state cell above
BUDGETS = [50, 100, 200]
samp2 = samp.select(["sn1", "n1", "a1", "m1", "fk"]).with_columns(samp[cc].alias("ct"))
countries = sorted(samp2["ct"].unique().to_list())
res = {b: {ct: [] for ct in countries} for b in BUDGETS}
t0 = time.time(); n = 0
for row in samp2.iter_rows(named=True):
    ev = build_keys_t(row)
    for b in BUDGETS:
        res[b][row["ct"]].append(len(run_greedy(ev, b)))
    n += 1
dt = time.time() - t0
print(f"rows {n} | loop {dt/60:.1f} min | per-row {dt/n*1000:.1f} ms")

print(f"\n{'budget':>6} | {'country':>7} | {'mean':>7} {'med':>5} {'p99':>6} {'max':>5} {'zero%':>6}")
for b in BUDGETS:
    allc = []
    for ct in countries:
        cs = np.array(res[b][ct]); allc.append(cs)
        print(f"{b:>6} | {ct:>7} | {cs.mean():7.1f} {np.median(cs):5.0f} "
              f"{np.percentile(cs,99):6.0f} {cs.max():5.0f} {float((cs==0).mean())*100:6.1f}")
    a = np.concatenate(allc)
    proj = sum(res[b][ct].__len__() and np.mean(res[b][ct]) * fc[ct] for ct in countries)
    print(f"{b:>6} | {'ALL':>7} | {a.mean():7.1f} {np.median(a):5.0f} {np.percentile(a,99):6.0f} "
          f"{a.max():5.0f} {float((a==0).mean())*100:6.1f} || proj pairs {proj/1e6:.0f}M "
          f"~{proj*20/1e9:.1f} GB | full-run est {dt/n*1732544/3600:.1f} h")

# --- cell 22 ---
PMAP = {"N": P_N, "A": P_A, "M": P_M}
assert PMAP["N"] is P_N and len(P_N) > 20000, "train postings missing - re-run the blocking-state cell first"
print("PMAP -> train postings:", len(P_N), len(P_A), len(P_M), "| EX:", len(EX))

