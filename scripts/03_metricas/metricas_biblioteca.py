"""
Metricas de uplift da biblioteca causalml, sobre todos os modelos.
 - qini_score / auuc_score : nao centradas em zero sob o nulo -> nulo por permutacao
 - rate_score (qini, autoc): traz erro-padrao e p-valor de fabrica
"""
import contextlib
import io as _io
import subprocess
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, qini_score, rate_score

SC = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\f728b89b-8274-4692-8ac2-0528394a10c3"
      r"\scratchpad")
N_PERM = 400
_buf = _io.StringIO()


def quieto(fn, *a, **k):
    with contextlib.redirect_stderr(_buf), contextlib.redirect_stdout(_buf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)


def ler_wsl(caminho, sep=None):
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"cat {caminho}"],
                       capture_output=True, text=True)
    if sep:
        return pd.read_csv(_io.StringIO(r.stdout), sep=sep, header=None, engine="python")
    return pd.read_csv(_io.StringIO(r.stdout), header=None)


d = ler_wsl("~/lbcf/Data/hillstrom/hillstrom_input.txt", sep=r"\s+")
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)

full = {}
for nome, cam in [("UDCF default", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip001"),
                  ("UDCF ip=0", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip0"),
                  ("Ablacao ip=0", "~/lbcf/Code/Model/LBCF/output/abl_abla_ip0")]:
    m = ler_wsl(cam)
    m.columns = ["mens", "womens"]
    full[nome] = m
for arq, rot in [("Chi", "Chi"), ("ED", "ED"), ("CTS", "CTS"),
                 ("Slearner", "S-learner"), ("Tlearner", "T-learner")]:
    m = pd.read_csv(f"{SC}\\baseline_{arq}.csv", header=None)
    m.columns = ["mens", "womens"]
    full[rot] = m
mbcf = {}
for ip, rot in [("0.01", "MBCF default"), ("0", "MBCF ip=0")]:
    mbcf[rot] = {a: ler_wsl(f"~/mbcf/out/mbcf_{a}_ip{ip}")[0].values.astype(float)
                 for a in ["mens", "womens"]}

for arm, t_arm in [("mens", Tm), ("womens", Tw)]:
    sel = np.where(ctrl | (t_arm == 1))[0]
    df = pd.DataFrame({"y": Y[sel], "w": t_arm[sel]})
    for nome, m in full.items():
        df[nome] = m[arm].values[sel]
    for nome in ["MBCF default", "MBCF ip=0"]:
        df[nome] = mbcf[nome][arm]
    modelos = [c for c in df.columns if c not in ("y", "w")]

    print()
    print("=" * 100)
    print(f"{arm.upper()}   n={len(df):,}   modelos={len(modelos)}")
    print("=" * 100)

    q_obs = quieto(qini_score, df, outcome_col="y", treatment_col="w", normalize=False)
    a_obs = quieto(auuc_score, df, outcome_col="y", treatment_col="w", normalize=False)
    r_q = quieto(rate_score, df, outcome_col="y", treatment_col="w",
                 weighting="qini", return_ci=True, n_bootstrap=200)
    r_a = quieto(rate_score, df, outcome_col="y", treatment_col="w",
                 weighting="autoc", return_ci=True, n_bootstrap=200)

    # nulo por permutacao para qini e auuc
    rng = np.random.default_rng(42)
    nq = {m: [] for m in modelos}
    na = {m: [] for m in modelos}
    for b in range(N_PERM):
        dp = df.copy()
        dp["w"] = rng.permutation(df["w"].values)
        qq = quieto(qini_score, dp, outcome_col="y", treatment_col="w", normalize=False)
        aa = quieto(auuc_score, dp, outcome_col="y", treatment_col="w", normalize=False)
        for m in modelos:
            if m in qq.index:
                nq[m].append(qq[m])
            if m in aa.index:
                na[m].append(aa[m])
    print(f"  ({N_PERM} permutacoes concluidas)")
    print()
    print(f"{'modelo':<16}{'QINI':>9}{'p perm':>9}{'AUUC':>9}{'p perm':>9}"
          f"{'RATE qini':>11}{'p':>8}{'RATE autoc':>12}{'p':>8}")
    print("-" * 100)
    for m in modelos:
        qv = q_obs.get(m, np.nan)
        av = a_obs.get(m, np.nan)
        pq = (np.array(nq[m]) >= qv).mean() if nq[m] else np.nan
        pa = (np.array(na[m]) >= av).mean() if na[m] else np.nan
        rq = r_q.loc[m] if m in r_q.index else None
        ra = r_a.loc[m] if m in r_a.index else None
        print(f"{m:<16}{qv:>9.4f}{pq:>9.3f}{av:>9.4f}{pa:>9.3f}"
              f"{rq['rate']:>11.6f}{rq['p_value']:>8.3f}"
              f"{ra['rate']:>12.6f}{ra['p_value']:>8.3f}")
