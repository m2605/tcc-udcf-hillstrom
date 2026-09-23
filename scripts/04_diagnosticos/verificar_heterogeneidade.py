"""
A heterogeneidade que aparece na Etapa 3 (imbalance_penalty=0) e sinal ou ruido?

Teste independente: comparar o CATE por subgrupo que o UDCF produziu com a
diferenca de taxas de conversao calculada DIRETAMENTE dos dados brutos, com
intervalo de confianca. Se o UDCF esta captando sinal, as duas coisas tem que
concordar. Nao depende de nada da algebra do criterio de split.
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")

# CATE por subgrupo que o UDCF (ip=0) produziu, reportado pela usuaria
UDCF_IP0 = {
    ("zip_code", "Rural"): 0.006079,
    ("zip_code", "Surburban"): 0.007144,
    ("zip_code", "Urban"): 0.006475,
    ("channel", "Multichannel"): 0.009441,
    ("channel", "Phone"): 0.005916,
    ("channel", "Web"): 0.006763,
    ("newbie", 0): 0.006217,
    ("newbie", 1): 0.007212,
}


def ate_bruto(sub, braco="Mens E-Mail"):
    """Diferenca de taxas de conversao com erro padrao (dois grupos independentes)."""
    t = sub.loc[sub["segment"] == braco, "conversion"]
    c = sub.loc[sub["segment"] == "No E-Mail", "conversion"]
    dif = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    z = dif / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
    return dif, se, p, len(t), len(c)


print("=" * 88)
print("CATE do UDCF (ip=0) vs. diferenca de medias calculada direto dos dados brutos")
print("Braco: Mens E-Mail vs No E-Mail")
print("=" * 88)
print(f"{'subgrupo':<26}{'UDCF':>11}{'bruto':>11}{'IC 95% bruto':>24}{'p':>9}{'bate?':>8}")
print("-" * 88)

ate_global, se_global, _, _, _ = ate_bruto(df)
linhas = []
for (col, val), udcf in UDCF_IP0.items():
    sub = df[df[col] == val]
    dif, se, p, nt, nc = ate_bruto(sub)
    lo, hi = dif - 1.96 * se, dif + 1.96 * se
    dentro = lo <= udcf <= hi
    linhas.append((col, val, udcf, dif, se, p, dentro))
    print(f"{f'{col}={val}':<26}{udcf:>11.6f}{dif:>11.6f}"
          f"{f'[{lo:+.5f}, {hi:+.5f}]':>24}{p:>9.3f}{'sim' if dentro else 'NAO':>8}")

print("-" * 88)
print(f"{'GLOBAL (todos)':<26}{'':>11}{ate_global:>11.6f}"
      f"{f'[{ate_global-1.96*se_global:+.5f}, {ate_global+1.96*se_global:+.5f}]':>24}")

print("\n" + "=" * 88)
print("As diferencas ENTRE subgrupos sao estatisticamente distinguiveis?")
print("(teste de interacao: o efeito do tratamento difere entre as categorias?)")
print("=" * 88)
for col in ["zip_code", "channel", "newbie"]:
    sub = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    sub["trat"] = (sub["segment"] == "Mens E-Mail").astype(float)
    # regressao: conversion ~ trat * categoria  -> teste F do termo de interacao
    cats = sorted(sub[col].unique())
    Xd = [np.ones(len(sub)), sub["trat"].values]
    nomes = ["intercepto", "trat"]
    for c in cats[1:]:
        d = (sub[col] == c).astype(float).values
        Xd.append(d)
        nomes.append(f"{col}={c}")
    base = len(Xd)
    for c in cats[1:]:
        d = (sub[col] == c).astype(float).values
        Xd.append(d * sub["trat"].values)
        nomes.append(f"trat x {col}={c}")
    Xd = np.column_stack(Xd)
    y = sub["conversion"].values
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n, k = Xd.shape
    sse_full = (resid ** 2).sum()
    Xr = Xd[:, :base]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    sse_red = ((y - Xr @ br) ** 2).sum()
    q = k - base
    F = ((sse_red - sse_full) / q) / (sse_full / (n - k))
    p_int = 1 - stats.f.cdf(F, q, n - k)
    print(f"  {col:<12} F({q},{n-k}) = {F:6.3f}   p = {p_int:.4f}   "
          f"{'INTERACAO SIGNIFICATIVA' if p_int < 0.05 else 'nao significativa'}")

print("\n" + "=" * 88)
print("Amplitude do CATE do UDCF vs. ruido amostral esperado")
print("=" * 88)
print(f"  desvio do CATE (ip=0), mens : 0.002734")
print(f"  erro padrao do ATE global   : {se_global:.6f}")
print(f"  razao                       : {0.002734/se_global:.2f}")
print("\n  Se a razao for << 1, a dispersao do CATE e menor que a incerteza do proprio")
print("  ATE global -> a 'heterogeneidade' pode ser indistinguivel de ruido.")
