"""
Por que o MBCF corta e o UDCF nao, com o MESMO limiar de 0.01?

Hipotese: os dois relabelings produzem pseudo-desfechos em escalas diferentes.
  UDCF        : rho = (W_c A^-1) * residual,  A = W'W ~ O(n)  ->  rho ~ 1/n
  Instrumental: rho = (Z - Zbar) * residual                   ->  rho ~ O(1)

Testa calculando os dois no MESMO no, com os mesmos dados.
"""
import io
import subprocess

import numpy as np
import pandas as pd

r = subprocess.run(["wsl", "-e", "bash", "-c",
                    "cat ~/lbcf/Data/hillstrom/hillstrom_input.txt"],
                   capture_output=True, text=True)
d = pd.read_csv(io.StringIO(r.stdout), sep=r"\s+", header=None)
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)


def rho_udcf(idx):
    """UDCFRelabelingStrategy: rho = (W_c A^-1) * residual"""
    y = Y[idx].reshape(-1, 1)
    W = np.column_stack([Tm[idx], Tw[idx]])
    yc = y - y.mean(0)
    Wc = W - W.mean(0)
    A = Wc.T @ Wc
    A_inv = np.linalg.inv(A)
    beta = A_inv @ Wc.T @ yc
    return (Wc @ A_inv.T) * (yc - Wc @ beta)


def rho_instrumental(idx):
    """InstrumentalRelabelingStrategy com instrumento = tratamento"""
    y = Y[idx]
    w = Tm[idx]           # braco mens vs controle
    z = w                 # instrumento = tratamento
    ybar, wbar, zbar = y.mean(), w.mean(), z.mean()
    num = ((z - zbar) * (y - ybar)).sum()
    den = ((z - zbar) * (w - wbar)).sum()
    late = num / den
    residual = (y - ybar) - late * (w - wbar)
    return ((z - zbar) * residual).reshape(-1, 1)


print("=" * 78)
print("ESCALA DOS PSEUDO-DESFECHOS, NO MESMO NO")
print("=" * 78)

# Cada modelo ve dados diferentes: o UDCF ve os 3 grupos (precisa dos K bracos
# para inverter W'W); o MBCF ve controle + um braco. Damos a cada um o seu.
sel_mens = np.where(ctrl | (Tm == 1))[0]
todos = np.arange(len(Y))
rng = np.random.default_rng(42)

print("  (UDCF: amostra com os 3 grupos | Instrumental: controle + mens)")
for n_no in [40000, 16000, 4000, 1000]:
    idx_u = rng.choice(todos, n_no, replace=False)
    idx_i = rng.choice(sel_mens, n_no, replace=False)
    ru = rho_udcf(idx_u)
    ri = rho_instrumental(idx_i)
    print(f"\n  no com n = {n_no:,}")
    print(f"    UDCF          |rho| medio = {np.abs(ru).mean():.3e}")
    print(f"    Instrumental  |rho| medio = {np.abs(ri).mean():.3e}")
    print(f"    razao instrumental/UDCF = {np.abs(ri).mean()/np.abs(ru).mean():,.0f}x"
          f"   (n = {n_no:,})")

print()
print("=" * 78)
print("CONSEQUENCIA: max ||soma_esq||^2 vs o limiar de 0.01")
print("=" * 78)
X = d.iloc[:, 0:11].values.astype(float)


def max_norma(idx, rho):
    n = len(idx)
    melhor = 0.0
    for j in range(X.shape[1]):
        o = np.argsort(X[idx, j], kind="mergesort")
        v = X[idx][o, j]
        fr = np.nonzero(np.diff(v) != 0)[0]
        if len(fr) == 0:
            continue
        cum = np.cumsum(rho[o], axis=0)[fr]
        melhor = max(melhor, float((cum ** 2).sum(axis=1).max()))
    return melhor


idx_u = rng.choice(todos, 16000, replace=False)
idx_i = rng.choice(sel_mens, 16000, replace=False)
mu = max_norma(idx_u, rho_udcf(idx_u))
mi = max_norma(idx_i, rho_instrumental(idx_i))
print(f"  no de 16.000 amostras (cada modelo com os dados que de fato ve):")
print(f"    UDCF         max||s||^2 = {mu:.4e}   > 0.01? {mu > 0.01}")
print(f"    Instrumental max||s||^2 = {mi:.4e}   > 0.01? {mi > 0.01}")
print(f"    razao = {mi/mu:.3e}")
