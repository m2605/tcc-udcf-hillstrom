"""Qual min_node_size torna a Etapa 0 viavel? E ele restringe a Hillstrom?"""
import numpy as np
import pandas as pd

ALPHA, LAMBDA = 0.05, 0.01
PATH = (r"C:\Users\m2292\Documents\tese-mba\lbcf_repo"
        r"\WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS-main\Data\RCT_data\train_data_UDCF.csv")
raw = pd.read_csv(PATH, sep=r"\s+", header=None)
Ya = raw.iloc[:, 14].values.reshape(-1, 1).astype(float)
Wa = raw.iloc[:, 15:22].values.astype(float)
Xa = raw.iloc[:, 0:14].values.astype(float)


def relabel(Y, W):
    Yc, Wc = Y - Y.mean(0), W - W.mean(0)
    WW = Wc.T @ Wc
    if abs(np.linalg.det(WW)) <= 1e-10:
        return None
    A = np.linalg.inv(WW)
    return (Wc @ A.T) * (Yc - Wc @ (A @ Wc.T @ Yc))


def avaliar(X, W, rho, mns):
    n = len(W)
    menor = (W < W.mean(0))
    n_menor_no = menor.sum(0)
    tam_min = ((W ** 2).sum(0) - W.sum(0) ** 2 / n) * ALPHA
    soma_w, soma_w2 = W.sum(0), (W ** 2).sum(0)
    n_ok, melhor = 0, 0.0
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="mergesort")
        v = X[o, j]
        fr = np.nonzero(np.diff(v) != 0)[0]
        if len(fr) == 0:
            continue
        ne = (fr + 1).astype(float)
        nd = n - ne
        cm = np.cumsum(menor[o], 0)[fr]
        cw = np.cumsum(W[o], 0)[fr]
        cw2 = np.cumsum(W[o] ** 2, 0)[fr]
        ok = (cm >= mns).all(1) & ((ne[:, None] - cm) >= mns).all(1)
        ok &= ((n_menor_no - cm) >= mns).all(1)
        ok &= ((nd[:, None] - n_menor_no + cm) >= mns).all(1)
        ok &= ((cw2 - cw ** 2 / ne[:, None]) >= tam_min).all(1)
        ok &= ((soma_w2 - cw2 - (soma_w - cw) ** 2 / nd[:, None]) >= tam_min).all(1)
        n_ok += int(ok.sum())
        if ok.any():
            cr = np.cumsum(rho[o], 0)[fr]
            melhor = max(melhor, float(np.where(ok, (cr ** 2).sum(1), -np.inf).max()))
    return n_ok, melhor


print("=" * 72)
print("AUTORES — no raiz de uma arvore (500 amostras), variando min_node_size")
print("=" * 72)
rng = np.random.default_rng(42)
idx = rng.choice(len(raw), 1000, replace=False)[:500]
rho = relabel(Ya[idx], Wa[idx])
for mns in [50, 30, 20, 10, 5, 1]:
    n_ok, melhor = avaliar(Xa[idx], Wa[idx], rho, mns)
    status = "-" if n_ok == 0 else f"max||s||^2={melhor:.3e} passa limiar? {melhor > LAMBDA}"
    print(f"  min_node_size={mns:>3}: splits viaveis={n_ok:>5}   {status}")

print("\n" + "=" * 72)
print("HILLSTROM — no raiz de uma arvore (16000 amostras), variando min_node_size")
print("  (para confirmar que min_node_size NAO e o fator limitante aqui)")
print("=" * 72)
hill = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")
cod = hill["segment"].map({"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}).values
Wh = np.column_stack([(cod == 1).astype(float), (cod == 2).astype(float)])
Yh = hill["conversion"].values.reshape(-1, 1).astype(float)
Xh = pd.concat([hill[["recency", "history", "mens", "womens", "newbie"]].astype(float),
                pd.get_dummies(hill["zip_code"], prefix="zip", dtype=float),
                pd.get_dummies(hill["channel"], prefix="channel", dtype=float)],
               axis=1).values.astype(float)
ih = rng.choice(len(hill), 32000, replace=False)[:16000]
rh = relabel(Yh[ih], Wh[ih])
print(f"  amostras por braco no no: {Wh[ih].sum(0).astype(int)}")
for mns in [50, 5]:
    n_ok, melhor = avaliar(Xh[ih], Wh[ih], rh, mns)
    print(f"  min_node_size={mns:>3}: splits viaveis={n_ok:>6}   max||s||^2={melhor:.3e}   "
          f"passa limiar {LAMBDA}? {melhor > LAMBDA}")
print("\n  -> na Hillstrom min_node_size nao restringe; o que bloqueia e o limiar.")
