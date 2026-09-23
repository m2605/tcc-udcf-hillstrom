"""
Diagnostico nos DADOS DOS AUTORES (train_data_UDCF.csv), com a configuracao
exata do main.cpp original deles: outcome_index=14, treatment_index={15..21},
num_treatments=7.

Roda o MESMO teste aplicado a Hillstrom:
   split existe  <=>  ||sum_left||^2 > imbalance_penalty (= 0.01)
"""
import numpy as np
import pandas as pd

IMBALANCE_PENALTY = 0.01
MIN_NODE_SIZE = 50
ALPHA = 0.05

PATH = (r"C:\Users\m2292\Documents\tese-mba\lbcf_repo"
        r"\WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS-main\Data\RCT_data\train_data_UDCF.csv")

raw = pd.read_csv(PATH, sep=r"\s+", header=None)
print("=" * 74)
print("DADOS DOS AUTORES - train_data_UDCF.csv")
print("=" * 74)
print(f"shape: {raw.shape}   (linhas x colunas)")

OUTCOME_IDX = 14
TREAT_IDX = [15, 16, 17, 18, 19, 20, 21]
COVAR_IDX = list(range(0, 14))

Y = raw.iloc[:, OUTCOME_IDX].values.reshape(-1, 1).astype(float)
W = raw.iloc[:, TREAT_IDX].values.astype(float)
Xv = raw.iloc[:, COVAR_IDX].values.astype(float)

print(f"\ncovariaveis: colunas {COVAR_IDX[0]}..{COVAR_IDX[-1]}  ({len(COVAR_IDX)} features)")
print(f"outcome:     coluna {OUTCOME_IDX}")
print(f"tratamentos: colunas {TREAT_IDX}  ({len(TREAT_IDX)} bracos)")

print("\n--- ESCALA DO OUTCOME (coluna 14) ---")
print(f"  media   = {Y.mean():.4f}")
print(f"  desvio  = {Y.std():.4f}")
print(f"  min/max = {Y.min():.4f} / {Y.max():.4f}")
print(f"  mediana = {np.median(Y):.4f}")
print(f"  % zeros = {(Y == 0).mean() * 100:.2f}%")
print(f"  binario? {set(np.unique(Y)) <= {0.0, 1.0}}")

print("\n--- ESTRUTURA DO TRATAMENTO ---")
soma_linha = W.sum(axis=1)
print(f"  soma das 7 colunas por linha - valores unicos: {np.unique(soma_linha)}")
print(f"  linhas com todas as 7 colunas = 0 (controle): {(soma_linha == 0).sum()}")
print(f"  linhas com exatamente uma = 1:                {(soma_linha == 1).sum()}")
print("  tamanho de cada braco:")
for k, c in enumerate(TREAT_IDX):
    print(f"    coluna {c}: n={int(W[:, k].sum())}  |  Y medio = {Y[W[:, k] == 1].mean():.2f}")
if (soma_linha == 0).sum() > 0:
    print(f"    CONTROLE  : n={int((soma_linha == 0).sum())}  |  Y medio = {Y[soma_linha == 0].mean():.2f}")

Wc_all = W - W.mean(axis=0)
WW_all = Wc_all.T @ Wc_all
det_all = np.linalg.det(WW_all)
print(f"\n  det(WW_bar) na base toda = {det_all:.6e}")
print(f"  relabel falha (det ~ 0, tol 1e-10)? {abs(det_all) <= 1e-10}")
print(f"  posto de W centrado = {np.linalg.matrix_rank(Wc_all)} de {W.shape[1]}")


def relabel(idx):
    Yc = Y[idx] - Y[idx].mean(axis=0)
    Wc = W[idx] - W[idx].mean(axis=0)
    WW = Wc.T @ Wc
    det = np.linalg.det(WW)
    if abs(det) <= 1e-10:
        return None, det
    A_inv = np.linalg.inv(WW)
    beta = A_inv @ Wc.T @ Yc
    rho = (Wc @ A_inv.T) * (Yc - Wc @ beta)
    return rho, det


def max_sumleft_sq(idx, rho, enforce=True):
    n = len(idx)
    Wn = W[idx]
    mean_w = Wn.mean(axis=0)
    small = (Wn < mean_w)
    num_node_small = small.sum(axis=0)
    size_node = (Wn ** 2).sum(axis=0) - Wn.sum(axis=0) ** 2 / n
    min_child_size = size_node * ALPHA
    sum_w_tot, sum_w2_tot = Wn.sum(axis=0), (Wn ** 2).sum(axis=0)

    best, info = 0.0, None
    for j in range(Xv.shape[1]):
        vals = Xv[idx, j]
        order = np.argsort(vals, kind="mergesort")
        v_s, rho_s, small_s, w_s = vals[order], rho[order], small[order], Wn[order]
        boundary = np.nonzero(np.diff(v_s) != 0)[0]
        if len(boundary) == 0:
            continue
        cum_rho = np.cumsum(rho_s, axis=0)
        cum_small = np.cumsum(small_s, axis=0)
        cum_w = np.cumsum(w_s, axis=0)
        cum_w2 = np.cumsum(w_s ** 2, axis=0)
        for b in boundary:
            n_left = b + 1
            n_right = n - n_left
            if enforce:
                nls = cum_small[b]
                if np.any(nls < MIN_NODE_SIZE) or np.any(n_left - nls < MIN_NODE_SIZE):
                    continue
                if np.any(num_node_small - nls < MIN_NODE_SIZE):
                    continue
                if np.any(n_right - num_node_small + nls < MIN_NODE_SIZE):
                    continue
                if np.any(cum_w2[b] - cum_w[b] ** 2 / n_left < min_child_size):
                    continue
                vr = sum_w2_tot - cum_w2[b] - (sum_w_tot - cum_w[b]) ** 2 / n_right
                if np.any(vr < min_child_size):
                    continue
            val = float((cum_rho[b] ** 2).sum())
            if val > best:
                best, info = val, (f"col{COVAR_IDX[j]}", float(v_s[b]), n_left, n_right)
    return best, info


print("\n" + "=" * 74)
print(f"TESTE DE SPLIT NOS DADOS DOS AUTORES: ||sum_left||^2 > {IMBALANCE_PENALTY} ?")
print("=" * 74)

idx = np.arange(len(raw))
rho, det = relabel(idx)
if rho is None:
    print(f"  relabel FALHOU no no raiz (det={det:.3e}) -> nenhum split possivel")
else:
    print(f"  sum_node (deve ser ~0): {np.abs(rho.sum(axis=0)).max():.3e}")
    print(f"  escala dos pseudo-outcomes: |rho| medio = {np.abs(rho).mean():.3e}, "
          f"max = {np.abs(rho).max():.3e}")
    for enforce, nome in [(True, "com restricoes"), (False, "SEM restricoes")]:
        best, info = max_sumleft_sq(idx, rho, enforce)
        print(f"\n  max ||sum_left||^2 ({nome}): {best:.6e}")
        print(f"     -> HA SPLIT? {best > IMBALANCE_PENALTY}   "
              f"(limiar = {IMBALANCE_PENALTY})")
        if best > IMBALANCE_PENALTY:
            print(f"     -> margem: {best / IMBALANCE_PENALTY:,.0f}x acima do limiar")
        if info:
            print(f"     melhor candidato: {info[0]} <= {info[1]} "
                  f"(n_left={info[2]}, n_right={info[3]})")

print("\n" + "=" * 74)
print("COMPARACAO LADO A LADO")
print("=" * 74)
hill = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")
print(f"{'':<28} {'AUTORES (RCT)':>18} {'HILLSTROM':>18}")
print(f"{'n amostras':<28} {len(raw):>18,} {len(hill):>18,}")
print(f"{'n bracos de tratamento':<28} {7:>18} {2:>18}")
print(f"{'outcome':<28} {'continuo (R$)':>18} {'binario 0/1':>18}")
print(f"{'media do outcome':<28} {Y.mean():>18.4f} {hill['conversion'].mean():>18.4f}")
print(f"{'desvio do outcome':<28} {Y.std():>18.4f} {hill['conversion'].std():>18.4f}")
