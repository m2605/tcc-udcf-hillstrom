"""Qini (nulo por permutacao) + GATES, desfecho = visit, por braco."""
import contextlib
import io as _io
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from causalml.metrics import qini_score

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
CV = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50"
      r"\scratchpad\mbcf_cv_visit")
N_PERM = 400
_b = _io.StringIO()


def q(fn, *a, **k):
    with contextlib.redirect_stderr(_b), contextlib.redirect_stdout(_b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)


d = pd.read_csv(os.path.join(RES, "hillstrom_input_visit.txt"), sep=r"\s+", header=None)
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)
n = len(Y)

mods = {}
for arq, rot in [("visit_udcf_default", "UDCF default"), ("visit_udcf_ip0", "UDCF ip=0"),
                 ("visit_abla_ip0", "Ablacao ip=0")]:
    mods[rot] = pd.read_csv(os.path.join(RES, arq), header=None, sep=r",\s*",
                            engine="python").values.astype(float)
for arq, rot in [("visit_Chi.csv", "Chi"), ("visit_ED.csv", "ED"), ("visit_CTS.csv", "CTS"),
                 ("visit_Slearner.csv", "S-learner"), ("visit_Tlearner.csv", "T-learner")]:
    mods[rot] = pd.read_csv(os.path.join(RES, arq), header=None).values.astype(float)
idx = np.load(os.path.join(CV, "idx_teste.npy"), allow_pickle=True)
M = np.full((n, 2), np.nan)
for j, arm in enumerate(["mens", "womens"]):
    for f in range(5):
        M[np.asarray(idx[f], dtype=int), j] = pd.read_csv(
            os.path.join(CV, f"pred_{arm}_{f+1}"), header=None)[0].values
mods["MBCF"] = M

for j, (arm, T) in enumerate([("mens", Tm), ("womens", Tw)]):
    sel = np.where(ctrl | (T == 1))[0]
    y, w = Y[sel], T[sel]
    ate = y[w == 1].mean() - y[w == 0].mean()
    df = pd.DataFrame({"y": y, "w": w})
    for nome, c in mods.items():
        df[nome] = c[sel, j]
    nomes = [c for c in df.columns if c not in ("y", "w")]

    print()
    print("=" * 96)
    print(f"{arm.upper()}   n={len(sel):,}   ATE ingenuo={ate*100:+.3f} pp")
    print("=" * 96)

    obs = q(qini_score, df, outcome_col="y", treatment_col="w", normalize=False)
    rng = np.random.default_rng(42)
    nul = {m: [] for m in nomes}
    for b in range(N_PERM):
        dp = df.copy()
        dp["w"] = rng.permutation(df["w"].values)
        qq = q(qini_score, dp, outcome_col="y", treatment_col="w", normalize=False)
        for m in nomes:
            nul[m].append(qq[m])

    print(f"{'modelo':<18}{'QINI':>10}{'p perm':>9}   {'GATES Q1..Q5 (pp)':<46}{'Q5-Q1':>9}{'p':>8}")
    print("-" * 96)
    for m in nomes:
        pred = df[m].values
        pv = float((np.array(nul[m]) >= obs[m]).mean())
        qt = pd.qcut(pred, 5, labels=False, duplicates="drop")
        g, se = [], []
        for k in sorted(np.unique(qt)):
            s = qt == k
            t_, c_ = y[s & (w == 1)], y[s & (w == 0)]
            g.append(t_.mean() - c_.mean())
            se.append(np.sqrt(t_.var(ddof=1)/len(t_) + c_.var(ddof=1)/len(c_)))
        dq = g[-1] - g[0]
        sq = np.sqrt(se[-1]**2 + se[0]**2)
        pq = 2 * (1 - stats.norm.cdf(abs(dq / sq)))
        gs = " ".join(f"{x*100:>7.2f}" for x in g)
        print(f"{m:<18}{obs[m]:>10.2f}{pv:>9.3f}   {gs:<46}{dq*100:>+8.2f}{pq:>8.3f}")
