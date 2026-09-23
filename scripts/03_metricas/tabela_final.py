"""
Tabela final: todos os modelos, mesmas linhas de avaliacao, mesmas metricas,
todas as predicoes fora-da-amostra.
"""
import io
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

SC = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\f728b89b-8274-4692-8ac2-0528394a10c3"
      r"\scratchpad")


def ler_wsl(caminho, sep=None):
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"cat {caminho}"],
                       capture_output=True, text=True)
    if sep:
        return pd.read_csv(io.StringIO(r.stdout), sep=sep, header=None, engine="python")
    return pd.read_csv(io.StringIO(r.stdout), header=None)


d = ler_wsl("~/lbcf/Data/hillstrom/hillstrom_input.txt", sep=r"\s+")
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)

# modelos com predicao sobre as 64.000 linhas
full = {}
for nome, cam in [("UDCF default", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip001"),
                  ("UDCF ip=0", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip0"),
                  ("Ablacao ip=0", "~/lbcf/Code/Model/LBCF/output/abl_abla_ip0")]:
    m = ler_wsl(cam)
    m.columns = ["mens", "womens"]
    full[nome] = m
for crit, rot in [("Chi", "Chi"), ("ED", "ED"), ("CTS", "CTS"),
                  ("Slearner", "S-learner"), ("Tlearner", "T-learner")]:
    m = pd.read_csv(f"{SC}\\baseline_{crit}.csv", header=None)
    m.columns = ["mens", "womens"]
    full[rot] = m

# MBCF: uma floresta por braco, predicoes so sobre as linhas daquele braco
mbcf = {}
for ip, rot in [("0.01", "MBCF default"), ("0", "MBCF ip=0")]:
    mbcf[rot] = {}
    for arm in ["mens", "womens"]:
        p = ler_wsl(f"~/mbcf/out/mbcf_{arm}_ip{ip}")
        mbcf[rot][arm] = p[0].values.astype(float)


def blp(y, w, pred):
    p = w.mean()
    c = pred - pred.mean()
    X = np.column_stack([np.ones(len(y)), w, (w - p) * c])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    Xe = X * r[:, None]
    inv = np.linalg.inv(X.T @ X)
    V = inv @ (Xe.T @ Xe) @ inv
    return b[2], float(np.sqrt(np.diag(V))[2])


def avaliar(pred, y, w):
    coef, se = blp(y, w, pred)
    pv = 2 * (1 - stats.norm.cdf(abs(coef / se)))
    q = pd.qcut(pred, 5, labels=False, duplicates="drop")
    ates = []
    for k in sorted(np.unique(q)):
        s = q == k
        t, c = y[s & (w == 1)], y[s & (w == 0)]
        dd = t.mean() - c.mean()
        ee = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        ates.append((dd, ee))
    dq = ates[-1][0] - ates[0][0]
    sq = np.sqrt(ates[-1][1] ** 2 + ates[0][1] ** 2)
    pq = 2 * (1 - stats.norm.cdf(abs(dq / sq)))
    rng = np.random.default_rng(0)
    nulos = np.array([blp(y, rng.permutation(w), pred)[0] for _ in range(300)])
    return coef, pv, dq, pq, float((np.abs(nulos) >= abs(coef)).mean()), [a[0] for a in ates]


for arm, t_arm in [("mens", Tm), ("womens", Tw)]:
    sel = np.where(ctrl | (t_arm == 1))[0]
    y = Y[sel]
    w = t_arm[sel]
    ate = y[w == 1].mean() - y[w == 0].mean()
    rot = "MENS E-MAIL" if arm == "mens" else "WOMENS E-MAIL"

    print()
    print("=" * 96)
    print(f"{rot}   n={len(sel):,}   ATE ingenuo={ate:.6f}")
    print("=" * 96)
    print(f"{'modelo':<16}{'splits':>9}{'media':>10}{'desvio':>10}"
          f"{'BLP':>9}{'p':>8}{'Q5-Q1':>11}{'p':>8}{'plac p':>9}")
    print("-" * 96)

    splits = {"UDCF default": 0, "UDCF ip=0": 11407, "Ablacao ip=0": 28375,
              "Chi": None, "ED": None, "CTS": None,
              "MBCF default": 19239 if arm == "mens" else 17105,
              "MBCF ip=0": 23615 if arm == "mens" else 21178}
    gates_todos = {}

    for nome, m in full.items():
        pred = m[arm].values[sel]
        coef, pv, dq, pq, pe, g = avaliar(pred, y, w)
        gates_todos[nome] = g
        sp = splits.get(nome)
        print(f"{nome:<16}{(str(sp) if sp is not None else '-'):>9}"
              f"{pred.mean():>10.6f}{pred.std():>10.6f}"
              f"{coef:>9.3f}{pv:>8.3f}{dq:>11.6f}{pq:>8.3f}{pe:>9.3f}")

    for nome in ["MBCF default", "MBCF ip=0"]:
        pred = mbcf[nome][arm]
        assert len(pred) == len(sel), f"{len(pred)} vs {len(sel)}"
        coef, pv, dq, pq, pe, g = avaliar(pred, y, w)
        gates_todos[nome] = g
        print(f"{nome:<16}{splits[nome]:>9}{pred.mean():>10.6f}{pred.std():>10.6f}"
              f"{coef:>9.3f}{pv:>8.3f}{dq:>11.6f}{pq:>8.3f}{pe:>9.3f}")

    print(f"\n  GATES — ATE bruto por quintil:")
    print(f"    {'modelo':<16}" + "".join(f"{'Q'+str(i+1):>11}" for i in range(5)))
    for nome, g in gates_todos.items():
        print(f"    {nome:<16}" + "".join(f"{x:>11.6f}" for x in g))
