"""
Avaliacao orientada a decisao de aplicacao.

Metrica 1 — VALOR DA POLITICA
  A politica de um modelo: para cada pessoa, escolher entre {nada, mens, womens}
  pelo maior CATE estimado (nada, se o maior for <= 0).
  O valor e estimado nas pessoas cujo braco SORTEADO coincidiu com o recomendado
  (estimador de Hajek; valido porque a atribuicao e aleatorizada e independente de X).

  Referencias:
    1. nao abordar ninguem
    2. mens para todos
    3. womens para todos
    4. braco sorteado para todos
    5. mesmas pessoas do modelo, braco sorteado  <- isola a camada 2

Metrica 2 — curva de cobertura: tratar os top q% pelo maior CATE
Metrica 3 — (Qini e GATES ficam em script separado)
"""
import os

import numpy as np
import pandas as pd

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
CV = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50"
      r"\scratchpad\mbcf_cv")
B_BOOT = 2000
rng_global = np.random.default_rng(7)

d = pd.read_csv(os.path.join(RES, "hillstrom_input.txt"), sep=r"\s+", header=None)
Y = d[11].values.astype(float)
Tm = d[12].values.astype(int)
Tw = d[13].values.astype(int)
ACAO = np.where(Tm == 1, 1, np.where(Tw == 1, 2, 0))  # 0=nada 1=mens 2=womens
n = len(Y)

print(f"n={n:,}  |  controle={int((ACAO==0).sum()):,}  "
      f"mens={int((ACAO==1).sum()):,}  womens={int((ACAO==2).sum()):,}")
print(f"taxas brutas: nada={Y[ACAO==0].mean()*100:.3f}%  "
      f"mens={Y[ACAO==1].mean()*100:.3f}%  womens={Y[ACAO==2].mean()*100:.3f}%")
print()

# ---------------- carrega os CATEs ----------------
modelos = {}


def dois_cols(caminho, sep=None):
    m = pd.read_csv(caminho, header=None, sep=sep, engine="python" if sep else "c")
    m.columns = ["mens", "womens"]
    return m[["mens", "womens"]].values.astype(float)


for nome, arq in [("UDCF default", "abl_udcf_ip001"), ("UDCF ip=0", "abl_udcf_ip0"),
                  ("Ablacao ip=0", "abl_abla_ip0")]:
    modelos[nome] = pd.read_csv(os.path.join(RES, arq), header=None,
                                sep=r",\s*", engine="python").values.astype(float)
for arq, rot in [("baseline_Chi.csv", "Chi"), ("baseline_ED.csv", "ED"),
                 ("baseline_CTS.csv", "CTS"), ("baseline_Slearner.csv", "S-learner"),
                 ("baseline_Tlearner.csv", "T-learner")]:
    modelos[rot] = pd.read_csv(os.path.join(RES, arq), header=None).values.astype(float)

# MBCF vindo da validacao cruzada
if os.path.exists(os.path.join(CV, "pred_mens_5")):
    idx = np.load(os.path.join(CV, "idx_teste.npy"), allow_pickle=True)
    M = np.full((n, 2), np.nan)
    for j, arm in enumerate(["mens", "womens"]):
        for f in range(5):
            p = pd.read_csv(os.path.join(CV, f"pred_{arm}_{f+1}"), header=None)
            M[np.asarray(idx[f], dtype=int), j] = p[0].values.astype(float)
    if not np.isnan(M).any():
        modelos["MBCF"] = M
        print("MBCF (validacao cruzada) carregado")
    else:
        print(f"MBCF incompleto: {int(np.isnan(M).sum())} faltantes")
else:
    print("MBCF ainda nao disponivel — sera omitido")
print(f"modelos carregados: {len(modelos)}")
print()


# ---------------- estimador de valor ----------------
def valor(recomendacao, idx=None):
    """Taxa de conversao entre quem recebeu o que a politica recomendaria."""
    a = ACAO if idx is None else ACAO[idx]
    y = Y if idx is None else Y[idx]
    r = recomendacao if idx is None else recomendacao[idx]
    casa = (a == r)
    if casa.sum() == 0:
        return np.nan, 0
    return y[casa].mean(), int(casa.sum())


def boot_ic(recomendacao, B=B_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, n)
        vals[b] = valor(recomendacao, s)[0]
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5), np.nanstd(vals)


def politica_modelo(cate):
    melhor = cate.argmax(axis=1) + 1          # 1=mens 2=womens
    trata = cate.max(axis=1) > 0
    return np.where(trata, melhor, 0)


# ---------------- referencias ----------------
print("=" * 88)
print("POLITICAS DE REFERENCIA")
print("=" * 88)
print(f"{'politica':<38}{'conversao':>11}{'IC 95%':>22}{'n casado':>12}")
print("-" * 88)

refs = {}
for rot, rec in [("1. nao abordar ninguem", np.zeros(n, dtype=int)),
                 ("2. mens para todos", np.ones(n, dtype=int)),
                 ("3. womens para todos", np.full(n, 2))]:
    v, nc = valor(rec)
    lo, hi, se = boot_ic(rec)
    refs[rot] = (v, lo, hi, se, nc)
    print(f"{rot:<38}{v*100:>10.3f}%  [{lo*100:>6.3f}%, {hi*100:>6.3f}%]{nc:>12,}")

# 4. braco sorteado para todos (media sobre sorteios)
vs = []
for r in range(200):
    rr = np.random.default_rng(1000 + r)
    rec = rr.integers(1, 3, n)
    vs.append(valor(rec)[0])
v4 = float(np.mean(vs))
aprox = "~" + format(n // 3, ",")
print(f"{'4. braco sorteado para todos':<38}{v4*100:>10.3f}%  "
      f"[{np.percentile(vs,2.5)*100:>6.3f}%, {np.percentile(vs,97.5)*100:>6.3f}%]"
      f"{aprox:>12}")
refs["4. braco sorteado"] = (v4, np.percentile(vs, 2.5), np.percentile(vs, 97.5), np.std(vs), n // 3)

# ---------------- politicas dos modelos ----------------
print()
print("=" * 88)
print("VALOR DA POLITICA DE CADA MODELO")
print("  (e a referencia 5: mesmas pessoas, braco sorteado)")
print("=" * 88)
v_mens_todos = refs["2. mens para todos"][0]
print(f"{'modelo':<16}{'%trat':>7}{'%mens':>7}{'conversao':>11}{'IC 95%':>22}"
      f"{'ref.5':>9}{'vs ref5':>9}{'vs mens':>9}")
print("-" * 88)

resultados = {}
for nome, cate in modelos.items():
    rec = politica_modelo(cate)
    v, nc = valor(rec)
    lo, hi, se = boot_ic(rec)
    tratados = rec > 0
    pct_mens = (rec[tratados] == 1).mean() if tratados.sum() else np.nan
    # referencia 5: mesmas pessoas tratadas, braco sorteado
    v5s = []
    for r in range(200):
        rr = np.random.default_rng(2000 + r)
        rec5 = np.where(tratados, rr.integers(1, 3, n), 0)
        v5s.append(valor(rec5)[0])
    v5 = float(np.mean(v5s))
    resultados[nome] = dict(v=v, lo=lo, hi=hi, se=se, n=nc, v5=v5,
                            frac=tratados.mean(), pct_mens=pct_mens)
    print(f"{nome:<16}{tratados.mean()*100:>6.1f}%{pct_mens*100:>6.1f}%"
          f"{v*100:>10.3f}%  [{lo*100:>6.3f}%, {hi*100:>6.3f}%]"
          f"{v5*100:>8.3f}%{(v-v5)*100:>+9.3f}{(v-v_mens_todos)*100:>+9.3f}")

print()
print("  '%mens'   = fracao dos tratados que o modelo manda para o braco masculino.")
print("  'vs ref5' = ganho sobre sortear o braco. CUIDADO: se %mens ~ 100%, esse ganho")
print("              e apenas o efeito marginal de mens > womens, NAO personalizacao.")
print("  'vs mens' = ganho sobre mandar mens para TODOS. Esta e a barra honesta")
print("              para a camada 2: so supera quem personaliza de verdade.")

# ---------------- curva de cobertura ----------------
print()
print("=" * 88)
print("CURVA DE COBERTURA — tratar os top q% pelo maior CATE estimado")
print("=" * 88)
qs = [0.1, 0.25, 0.5, 0.75, 1.0]
print(f"{'modelo':<16}" + "".join(f"{'q='+str(int(q*100))+'%':>12}" for q in qs))
print("-" * 88)
for nome, cate in modelos.items():
    melhor_val = cate.max(axis=1)
    melhor_arm = cate.argmax(axis=1) + 1
    linha = f"{nome:<16}"
    for q in qs:
        k = int(n * q)
        corte = np.sort(melhor_val)[::-1][k - 1] if k > 0 else np.inf
        rec = np.where(melhor_val >= corte, melhor_arm, 0)
        v, _ = valor(rec)
        linha += f"{v*100:>11.3f}%"
    print(linha)
print()
print(f"  referencia: nao abordar ninguem = {refs['1. nao abordar ninguem'][0]*100:.3f}%")
print(f"              mens para todos     = {refs['2. mens para todos'][0]*100:.3f}%")
