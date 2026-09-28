"""features.py - shared feature core for inference.py AND train.py; FEATS order is also
the X.npy column contract that rescore.py consumes.
Features 0..19 are copied VERBATIM from baseline-0.924/inference.py (see ../../docs/methodology.md).
Features 20..35 are new in v4. FEATS order is the train/test contract.
"""
import re
import unicodedata
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

PUNCT = re.compile(r"[!-/:-@\[-`{-~]")
RE2 = re.compile(r"\d{2,}")
RE3 = re.compile(r"\d{3,}")
RENUM = re.compile(r"\d{2,}")

def n2(s):
    s = s if isinstance(s, str) else ""
    return re.sub(r"\s+", " ", PUNCT.sub("", s.upper())).strip()

SUF2 = re.compile(r"\b(LLC|INC|CORP(?:ORATION)?|LTD|LIMITED|PVT|PRIVATE|LLP|PLC|LP|CO|COMPANY|SAS|SARL|SASU|EURL|SA|SCI)\b", re.I)
def strip2(s):
    return re.sub(r"\s+", " ", SUF2.sub("", s)).strip()

def foldp(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))

def sclass(s):
    if s.isascii():
        return 0
    names = {unicodedata.name(c, "") for c in s if not c.isascii()}
    return 1 if names and all(x.startswith("LATIN") for x in names) else 2

def _grams(s, n):
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

OLD = ["nratio", "nratio_suf", "nexact", "nexact_suf", "nexact_fold", "pref4_eq",
       "tok_cont", "tok_jacc", "acont", "ajacc", "addr_exact", "addr_missing",
       "shared_num2", "shared_num3", "both_num", "sc_A", "sc_B", "sc_diff",
       "lendiff", "country_eq"]
NEW = ["n_jw", "n_partial", "n_token_sort", "n_token_set",
       "a_jw", "a_token_sort", "a_partial",
       "n_char2_jacc", "n_char3_jacc", "first_tok_eq",
       "addr_pref2_eq", "addr_pref5_eq",
       "harm_nr_aj", "min_nr_aj", "lognum_sim", "a_lendiff"]
FEATS = OLD + NEW
NF = len(FEATS)
assert NF == 36, NF

def featurize(na_r, nb_r, aa_r, ab_r, cta, ctb):
    """Raw strings in, 36 float32 features out. Identical code on train and test."""
    na, nb = n2(na_r), n2(nb_r)
    aa, ab = n2(aa_r), n2(ab_r)
    x = np.zeros(NF, dtype=np.float32)
    ta, tb = set(na.split()), set(nb.split())
    ua, ub = set(aa.split()), set(ab.split())
    sa, sb = set(RE2.findall(aa)), set(RE2.findall(ab))
    na_s, nb_s = strip2(na), strip2(nb)
    it, iau = len(ta & tb), len(ua & ub)
    x[0] = fuzz.ratio(na, nb) / 100
    x[1] = fuzz.ratio(na_s, nb_s) / 100
    x[2] = float(bool(na) and na == nb)
    x[3] = float(bool(na_s) and na_s == nb_s)
    x[4] = float(bool(na) and foldp(na) == foldp(nb))
    x[5] = float(len(na) >= 4 and len(nb) >= 4 and na[:4] == nb[:4])
    x[6] = it / len(ta) if ta else 0.0
    x[7] = it / len(ta | tb) if (ta | tb) else 0.0
    x[8] = iau / min(len(ua), len(ub)) if ua and ub else 0.0
    x[9] = iau / len(ua | ub) if (ua | ub) else 0.0
    x[10] = float(bool(aa) and aa == ab)
    x[11] = float(not aa or not ab)
    x[12] = float(bool(sa & sb))
    x[13] = float(bool(set(RE3.findall(aa)) & set(RE3.findall(ab))))
    x[14] = float(bool(sa) and bool(sb))
    sA, sB = sclass(na), sclass(nb)
    x[15], x[16] = float(sA), float(sB)
    x[17] = float(sA != sB)
    x[18] = abs(len(na) - len(nb)) / (len(na) + len(nb) + 1)
    x[19] = float(cta == ctb)
    # ---- new in v4 ----
    x[20] = JaroWinkler.similarity(na, nb)
    x[21] = fuzz.partial_ratio(na, nb) / 100
    x[22] = fuzz.token_sort_ratio(na, nb) / 100
    x[23] = fuzz.token_set_ratio(na, nb) / 100
    x[24] = JaroWinkler.similarity(aa, ab)
    x[25] = fuzz.token_sort_ratio(aa, ab) / 100
    x[26] = fuzz.partial_ratio(aa, ab) / 100
    g2a, g2b = _grams(na, 2), _grams(nb, 2)
    x[27] = len(g2a & g2b) / len(g2a | g2b) if (g2a | g2b) else 0.0
    g3a, g3b = _grams(na, 3), _grams(nb, 3)
    x[28] = len(g3a & g3b) / len(g3a | g3b) if (g3a | g3b) else 0.0
    xa, xb = na.split(), nb.split()
    x[29] = float(bool(xa) and bool(xb) and xa[0] == xb[0])
    x[30] = float(len(aa) >= 2 and len(ab) >= 2 and aa[:2] == ab[:2])
    x[31] = float(len(aa) >= 5 and len(ab) >= 5 and aa[:5] == ab[:5])
    r0, a0 = float(x[0]), float(x[9])
    x[32] = (2.0 * r0 * a0 / (r0 + a0)) if (r0 + a0) > 0 else 0.0
    x[33] = min(r0, a0)
    ma, mb = RENUM.search(aa), RENUM.search(ab)
    if ma and mb:
        va, vb = np.log1p(float(ma.group())), np.log1p(float(mb.group()))
        x[34] = min(va, vb) / max(va, vb)
    else:
        x[34] = 0.0
    x[35] = abs(len(aa) - len(ab)) / (len(aa) + len(ab) + 1)
    return x