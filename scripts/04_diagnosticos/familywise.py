"""
Teste family-wise correto: qual a distribuicao nula do MENOR p-valor entre os 18
testes, dada a correlacao real entre os modelos?

Procedimento: permuta o tratamento, recalcula os 18 BLP, guarda o menor p.
As predicoes ficam fixas (foram ajustadas com o tratamento verdadeiro); permutar w
testa exatamente a hipotese de que elas nao carregam informacao sobre o efeito.
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

full = {}
for nome, cam in [("UDCF ip=0", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip0"),
                  ("Ablacao", "~/lbcf/Code/Model/LBCF/output/abl_abla_ip0")]:
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


def blp_p(y, w, pred):
    p = w.mean()
    c = pred - pred.mean()
    X = np.column_stack([np.ones(len(y)), w, (w - p) * c])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    Xe = X * r[:, None]
    inv = np.linalg.inv(X.T @ X)
    V = inv @ (Xe.T @ Xe) @ inv
    se = float(np.sqrt(np.diag(V))[2])
    return 2 * (1 - stats.norm.cdf(abs(b[2] / se)))


# monta a lista dos 18 testes: (y, w, pred)
testes = []
nomes = []
for arm, t_arm in [("mens", Tm), ("womens", Tw)]:
    sel = np.where(ctrl | (t_arm == 1))[0]
    y = Y[sel]
    w = t_arm[sel]
    for nome, m in full.items():
        testes.append((y, w, m[arm].values[sel]))
        nomes.append(f"{nome} [{arm}]")
    for nome in ["MBCF default", "MBCF ip=0"]:
        testes.append((y, w, mbcf[nome][arm]))
        nomes.append(f"{nome} [{arm}]")

print(f"numero de testes: {len(testes)}")
obs = np.array([blp_p(y, w, p) for y, w, p in testes])
i_min = int(np.argmin(obs))
print(f"menor p observado: {obs[i_min]:.4f}  ({nomes[i_min]})")
print()

print("=== correlacao entre as predicoes (braco womens) ===")
cols = {}
for nome, m in full.items():
    cols[nome] = m["womens"].values[np.where(ctrl | (Tw == 1))[0]]
for nome in ["MBCF default", "MBCF ip=0"]:
    cols[nome] = mbcf[nome]["womens"]
C = pd.DataFrame(cols).corr()
tri = C.values[np.triu_indices_from(C.values, 1)]
print(f"  correlacao media entre modelos: {tri.mean():.3f}")
print(f"  minima={tri.min():.3f}  maxima={tri.max():.3f}")
print()

print("=== distribuicao nula do MENOR p-valor (1000 permutacoes) ===")
rng = np.random.default_rng(42)
minimos = []
for b in range(1000):
    ps = []
    for y, w, pred in testes:
        ps.append(blp_p(y, rng.permutation(w), pred))
    minimos.append(min(ps))
    if (b + 1) % 250 == 0:
        print(f"  {b+1}/1000")
minimos = np.array(minimos)
print()
print(f"  menor p esperado sob o nulo (media) : {minimos.mean():.4f}")
print(f"  mediana                             : {np.median(minimos):.4f}")
print(f"  P(minimo <= {obs[i_min]:.4f}) sob o nulo  : {(minimos <= obs[i_min]).mean():.4f}")
print()
print(f"  para referencia, 18 testes independentes dariam ~0.58")
print(f"  e 18 testes perfeitamente correlacionados dariam ~{obs[i_min]:.3f}")
