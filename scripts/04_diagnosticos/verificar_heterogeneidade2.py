"""
Os testes de interacao anteriores cobriram zip_code, channel e newbie — que sao
justamente onde a floresta MENOS divide. A Etapa 3 divide 30% em `history` e 20%
em `recency`. Entao a ausencia de interacao naquelas tres variaveis nao diz nada
sobre heterogeneidade nestas duas.

Aqui testamos interacao tratamento x history e tratamento x recency.
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")
sub = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
sub["trat"] = (sub["segment"] == "Mens E-Mail").astype(float)
y = sub["conversion"].values


def teste_F(Xd, base_cols, rotulo, q):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    sse_full = ((y - Xd @ beta) ** 2).sum()
    Xr = Xd[:, :base_cols]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    sse_red = ((y - Xr @ br) ** 2).sum()
    n, k = Xd.shape
    F = ((sse_red - sse_full) / q) / (sse_full / (n - k))
    p = 1 - stats.f.cdf(F, q, n - k)
    print(f"  {rotulo:<42} F({q},{n-k}) = {F:6.3f}   p = {p:.4f}   "
          f"{'SIGNIFICATIVA' if p < 0.05 else 'nao significativa'}")
    return p


print("=" * 86)
print("INTERACAO COM AS VARIAVEIS ONDE A FLORESTA MAIS DIVIDE")
print("  (history = 30% dos splits, recency = 20%)")
print("=" * 86)

for var in ["history", "recency"]:
    x = sub[var].values.astype(float)
    xz = (x - x.mean()) / x.std()
    Xd = np.column_stack([np.ones(len(sub)), sub["trat"], xz, xz * sub["trat"]])
    teste_F(Xd, 3, f"trat x {var} (linear)", 1)

# versao nao-linear: quartis
print()
for var in ["history", "recency"]:
    q4 = pd.qcut(sub[var], 4, labels=False, duplicates="drop")
    cats = sorted(pd.unique(q4))
    cols = [np.ones(len(sub)), sub["trat"].values]
    for c in cats[1:]:
        cols.append((q4 == c).astype(float).values)
    base = len(cols)
    for c in cats[1:]:
        cols.append((q4 == c).astype(float).values * sub["trat"].values)
    Xd = np.column_stack(cols)
    teste_F(Xd, base, f"trat x {var} (quartis)", len(cats) - 1)

print("\n" + "=" * 86)
print("TESTE CONJUNTO: todas as 11 covariaveis interagindo com o tratamento")
print("=" * 86)
Xfeat = pd.concat([
    sub[["recency", "history", "mens", "womens", "newbie"]].astype(float),
    pd.get_dummies(sub["zip_code"], prefix="zip", dtype=float).iloc[:, 1:],
    pd.get_dummies(sub["channel"], prefix="channel", dtype=float).iloc[:, 1:],
], axis=1)
Xf = ((Xfeat - Xfeat.mean()) / Xfeat.std().replace(0, 1)).values
cols = [np.ones(len(sub)), sub["trat"].values] + [Xf[:, j] for j in range(Xf.shape[1])]
base = len(cols)
cols += [Xf[:, j] * sub["trat"].values for j in range(Xf.shape[1])]
Xd = np.column_stack(cols)
teste_F(Xd, base, "todas as interacoes conjuntamente", Xf.shape[1])

print("\n" + "=" * 86)
print("ATE BRUTO POR QUARTIL (olhando o padrao, nao so o teste)")
print("=" * 86)
for var in ["history", "recency"]:
    print(f"\n  {var}:")
    q4 = pd.qcut(sub[var], 4, labels=False, duplicates="drop")
    for c in sorted(pd.unique(q4)):
        s = sub[q4 == c]
        t = s.loc[s["trat"] == 1, "conversion"]
        ct = s.loc[s["trat"] == 0, "conversion"]
        d = t.mean() - ct.mean()
        se = np.sqrt(t.var(ddof=1) / len(t) + ct.var(ddof=1) / len(ct))
        faixa = f"[{s[var].min():.0f}, {s[var].max():.0f}]"
        print(f"    Q{c+1} {faixa:<18} ATE = {d:+.6f}  IC95% "
              f"[{d-1.96*se:+.6f}, {d+1.96*se:+.6f}]  n={len(s):,}")
