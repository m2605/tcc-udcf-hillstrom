"""
Diagnostico (NAO e reimplementacao do UDCF): mede a magnitude da quantidade
que o codigo dos autores usa para decidir se um split existe.

Do UDCFSplittingRule.cpp (linhas 413-420), com pesos = 1:
  decrease = ||sum_left||^2 / n_left + ||sum_node - sum_left||^2 / n_right
             - imbalance_penalty * (1/n_left + 1/n_right)
  aceita o candidato se decrease > 0.

Pela condicao de 1a ordem do OLS no relabeling, sum_node == 0, logo:
  decrease = (1/n_left + 1/n_right) * (||sum_left||^2 - imbalance_penalty)
  => decrease > 0  <=>  ||sum_left||^2 > imbalance_penalty (= 0.01)

Este script calcula max ||sum_left||^2 sobre todos os splits possiveis.
"""
import numpy as np
import pandas as pd

IMBALANCE_PENALTY = 0.01
MIN_NODE_SIZE = 50
ALPHA = 0.05

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")

Y = df["conversion"].astype(float).values.reshape(-1, 1)
code = df["segment"].map({"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}).values
W = np.column_stack([(code == 1).astype(float), (code == 2).astype(float)])

Xnum = df[["recency", "history", "mens", "womens", "newbie"]].astype(float)
Xzip = pd.get_dummies(df["zip_code"], prefix="zip", dtype=float)
Xch = pd.get_dummies(df["channel"], prefix="channel", dtype=float)
X = pd.concat([Xnum, Xzip, Xch], axis=1)
feat_names = list(X.columns)
Xv = X.values


def relabel(idx):
    """UDCFRelabelingStrategy::relabel, pesos = 1."""
    Yc = Y[idx] - Y[idx].mean(axis=0)
    Wc = W[idx] - W[idx].mean(axis=0)
    WW = Wc.T @ Wc
    det = np.linalg.det(WW)
    A_inv = np.linalg.inv(WW)
    beta = A_inv @ Wc.T @ Yc
    rho_weight = Wc @ A_inv.T
    residual = Yc - Wc @ beta
    rho = rho_weight * residual  # [n x 2], pois num_outcomes = 1
    return rho, det


def max_sumleft_sq(idx, rho, enforce_constraints=True):
    """Maior ||sum_left||^2 sobre todos os splits possiveis do no."""
    n = len(idx)
    Wn = W[idx]
    mean_w = Wn.mean(axis=0)
    # num_small_w[k] = # amostras com W_k abaixo da media do no  (=> W_k == 0)
    small = (Wn < mean_w)
    num_node_small = small.sum(axis=0)
    size_node = (Wn ** 2).sum(axis=0) - Wn.sum(axis=0) ** 2 / n
    min_child_size = size_node * ALPHA

    best = 0.0
    best_info = None
    for j in range(Xv.shape[1]):
        vals = Xv[idx, j]
        order = np.argsort(vals, kind="mergesort")
        v_sorted = vals[order]
        rho_sorted = rho[order]
        small_sorted = small[order]
        w_sorted = Wn[order]

        # fronteiras entre valores unicos
        boundary = np.nonzero(np.diff(v_sorted) != 0)[0]
        if len(boundary) == 0:
            continue

        cum_rho = np.cumsum(rho_sorted, axis=0)
        cum_small = np.cumsum(small_sorted, axis=0)
        cum_w = np.cumsum(w_sorted, axis=0)
        cum_w2 = np.cumsum(w_sorted ** 2, axis=0)

        for b in boundary:
            n_left = b + 1
            n_right = n - n_left
            if enforce_constraints:
                nls = cum_small[b]
                if np.any(nls < MIN_NODE_SIZE) or np.any(n_left - nls < MIN_NODE_SIZE):
                    continue
                if np.any(num_node_small - nls < MIN_NODE_SIZE):
                    continue
                if np.any(n_right - num_node_small + nls < MIN_NODE_SIZE):
                    continue
                var_left = cum_w2[b] - cum_w[b] ** 2 / n_left
                if np.any(var_left < min_child_size):
                    continue
                sum_w_tot = w_sorted.sum(axis=0)
                sum_w2_tot = (w_sorted ** 2).sum(axis=0)
                var_right = sum_w2_tot - cum_w2[b] - (sum_w_tot - cum_w[b]) ** 2 / n_right
                if np.any(var_right < min_child_size):
                    continue
            s = cum_rho[b]
            val = float((s ** 2).sum())
            if val > best:
                best = val
                best_info = (feat_names[j], float(v_sorted[b]), n_left, n_right)
    return best, best_info


rng = np.random.default_rng(42)
N = len(df)

print("=" * 72)
print("CONDICAO DE SPLIT (codigo dos autores):  ||sum_left||^2 > imbalance_penalty")
print(f"imbalance_penalty (default dos autores) = {IMBALANCE_PENALTY}")
print("=" * 72)

for label, idx in [
    ("Base completa (n=64000)", np.arange(N)),
    ("No raiz de 1 arvore, honesty (n=16000)", rng.choice(N, 16000, replace=False)),
    ("No raiz de 1 arvore, honesty (n=16000) - 2a amostra", rng.choice(N, 16000, replace=False)),
]:
    rho, det = relabel(idx)
    soma = rho.sum(axis=0)
    print(f"\n--- {label} ---")
    print(f"  det(WW_bar) = {det:.4e}   (limiar de falha do relabel: 1e-10)")
    print(f"  sum_node (deve ser ~0):        {soma}")
    print(f"  escala dos pseudo-outcomes: |rho| medio = {np.abs(rho).mean():.3e}, "
          f"max = {np.abs(rho).max():.3e}")
    for enforce, nome in [(True, "com restricoes (min_node_size, alpha)"),
                          (False, "SEM restricoes (limite superior teorico)")]:
        best, info = max_sumleft_sq(idx, rho, enforce)
        passa = best > IMBALANCE_PENALTY
        print(f"  max ||sum_left||^2 {nome}: {best:.6e}")
        print(f"      -> passa no teste (> {IMBALANCE_PENALTY})? {passa}")
        if info:
            print(f"      melhor candidato: {info[0]} <= {info[1]}  (n_left={info[2]}, n_right={info[3]})")
        if not passa and best > 0:
            c = np.sqrt(IMBALANCE_PENALTY / best)
            print(f"      fator de escala em Y necessario para haver split: {c:.1f}x")
