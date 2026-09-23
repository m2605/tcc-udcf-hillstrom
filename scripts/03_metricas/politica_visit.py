"""Valor da politica e curva de cobertura, desfecho = visit."""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
CV = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50"
      r"\scratchpad\mbcf_cv_visit")

d = pd.read_csv(os.path.join(RES, "hillstrom_input_visit.txt"), sep=r"\s+", header=None)
Y = d[11].values.astype(float)
Tm = d[12].values.astype(int)
Tw = d[13].values.astype(int)
ACAO = np.where(Tm == 1, 1, np.where(Tw == 1, 2, 0))
n = len(Y)

print(f"desfecho = visit | n={n:,}")
print(f"taxas: nada={Y[ACAO==0].mean()*100:.3f}%  mens={Y[ACAO==1].mean()*100:.3f}%  "
      f"womens={Y[ACAO==2].mean()*100:.3f}%")
print()

modelos = {}
for arq, rot in [("visit_udcf_default", "UDCF default"), ("visit_udcf_ip0", "UDCF ip=0"),
                 ("visit_abla_default", "Ablacao default"), ("visit_abla_ip0", "Ablacao ip=0")]:
    p = os.path.join(RES, arq)
    if os.path.exists(p):
        modelos[rot] = pd.read_csv(p, header=None, sep=r",\s*",
                                   engine="python").values.astype(float)
for arq, rot in [("visit_Slearner.csv", "S-learner"), ("visit_Tlearner.csv", "T-learner"),
                 ("visit_Chi.csv", "Chi"), ("visit_ED.csv", "ED"), ("visit_CTS.csv", "CTS")]:
    p = os.path.join(RES, arq)
    if os.path.exists(p):
        modelos[rot] = pd.read_csv(p, header=None).values.astype(float)

if os.path.exists(os.path.join(CV, "pred_womens_5")):
    idx = np.load(os.path.join(CV, "idx_teste.npy"), allow_pickle=True)
    M = np.full((n, 2), np.nan)
    ok = True
    for j, arm in enumerate(["mens", "womens"]):
        for f in range(5):
            fp = os.path.join(CV, f"pred_{arm}_{f+1}")
            if not os.path.exists(fp):
                ok = False
                break
            M[np.asarray(idx[f], dtype=int), j] = pd.read_csv(fp, header=None)[0].values
    if ok and not np.isnan(M).any():
        modelos["MBCF"] = M

print(f"modelos: {len(modelos)} -> {', '.join(modelos)}")
print()


def valor(rec, sub=None):
    a, y, r = (ACAO, Y, rec) if sub is None else (ACAO[sub], Y[sub], rec[sub])
    casa = (a == r)
    return (y[casa].mean(), int(casa.sum())) if casa.sum() else (np.nan, 0)


def boot(rec, B=1500, seed=7):
    rng = np.random.default_rng(seed)
    v = np.array([valor(rec, rng.integers(0, n, n))[0] for _ in range(B)])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5), np.nanstd(v)


print("=" * 92)
print("REFERENCIAS")
print("=" * 92)
refs = {}
for rot, rec in [("nao abordar ninguem", np.zeros(n, int)),
                 ("mens para todos", np.ones(n, int)),
                 ("womens para todos", np.full(n, 2))]:
    v, nc = valor(rec)
    lo, hi, se = boot(rec)
    refs[rot] = v
    print(f"  {rot:<24}{v*100:>8.3f}%  [{lo*100:>6.3f}%, {hi*100:>6.3f}%]  se={se*100:.3f}pp")
v_bar = refs["mens para todos"]

print()
print("=" * 92)
print("VALOR DA POLITICA DE CADA MODELO")
print("=" * 92)
print(f"{'modelo':<18}{'%trat':>7}{'%mens':>7}{'conversao':>11}{'IC 95%':>22}{'vs mens':>10}")
print("-" * 92)
guardado = {}
for nome, cate in modelos.items():
    melhor = cate.argmax(1) + 1
    trata = cate.max(1) > 0
    rec = np.where(trata, melhor, 0)
    v, nc = valor(rec)
    lo, hi, se = boot(rec)
    pm = (rec[trata] == 1).mean() if trata.sum() else np.nan
    guardado[nome] = (v, se)
    print(f"{nome:<18}{trata.mean()*100:>6.1f}%{pm*100:>6.1f}%{v*100:>10.3f}%  "
          f"[{lo*100:>6.3f}%, {hi*100:>6.3f}%]{(v-v_bar)*100:>+9.3f}")

print()
print(f"  barra = mens para todos = {v_bar*100:.3f}%")
print("  'vs mens' em pontos percentuais. Compare com o erro-padrao acima.")

print()
print("=" * 92)
print("CURVA DE COBERTURA — tratar os top q% pelo maior CATE")
print("=" * 92)
qs = [0.1, 0.25, 0.5, 0.75, 1.0]
print(f"{'modelo':<18}" + "".join(f"{'q='+str(int(q*100))+'%':>12}" for q in qs))
print("-" * 92)
for nome, cate in modelos.items():
    mv = cate.max(1)
    ma = cate.argmax(1) + 1
    linha = f"{nome:<18}"
    for q in qs:
        k = max(int(n * q), 1)
        corte = np.sort(mv)[::-1][k - 1]
        rec = np.where(mv >= corte, ma, 0)
        linha += f"{valor(rec)[0]*100:>11.3f}%"
    print(linha)
print(f"\n  nao abordar ninguem = {refs['nao abordar ninguem']*100:.3f}%")
