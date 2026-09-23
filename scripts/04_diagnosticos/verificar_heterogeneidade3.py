"""
Hipotese substantiva do dataset Hillstrom: o e-mail masculino deveria funcionar
melhor em quem tem historico de compra masculina, e o feminino em quem tem
historico feminino. E a heterogeneidade teoricamente esperada nesta base.
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")


def ate(sub, braco):
    t = sub.loc[sub["segment"] == braco, "conversion"]
    c = sub.loc[sub["segment"] == "No E-Mail", "conversion"]
    d = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return d, se, len(t), len(c)


print("=" * 84)
print("CORRESPONDENCIA ENTRE BRACO E HISTORICO DE COMPRA")
print("=" * 84)
for braco, var in [("Mens E-Mail", "mens"), ("Womens E-Mail", "womens")]:
    print(f"\n{braco}  x  historico `{var}`")
    ds = {}
    for v in [0, 1]:
        d, se, nt, nc = ate(df[df[var] == v], braco)
        ds[v] = (d, se)
        print(f"  {var}={v}: ATE = {d:+.6f}  IC95% [{d-1.96*se:+.6f}, {d+1.96*se:+.6f}]"
              f"  (n_trat={nt:,}, n_ctrl={nc:,})")
    dif = ds[1][0] - ds[0][0]
    se_dif = np.sqrt(ds[1][1] ** 2 + ds[0][1] ** 2)
    z = dif / se_dif
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"  diferenca (={var}=1 menos {var}=0): {dif:+.6f}  se={se_dif:.6f}  "
          f"z={z:+.2f}  p={p:.4f}   {'SIGNIFICATIVA' if p < 0.05 else 'nao significativa'}")

print("\n" + "=" * 84)
print("MESMA COISA PARA `visit` (desfecho menos raro, mais poder estatistico)")
print("=" * 84)


def ate_v(sub, braco, col):
    t = sub.loc[sub["segment"] == braco, col]
    c = sub.loc[sub["segment"] == "No E-Mail", col]
    d = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return d, se


for braco, var in [("Mens E-Mail", "mens"), ("Womens E-Mail", "womens")]:
    print(f"\n{braco}  x  `{var}`   (desfecho = visit)")
    ds = {}
    for v in [0, 1]:
        d, se = ate_v(df[df[var] == v], braco, "visit")
        ds[v] = (d, se)
        print(f"  {var}={v}: ATE = {d:+.6f}  IC95% [{d-1.96*se:+.6f}, {d+1.96*se:+.6f}]")
    dif = ds[1][0] - ds[0][0]
    se_dif = np.sqrt(ds[1][1] ** 2 + ds[0][1] ** 2)
    z = dif / se_dif
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"  diferenca: {dif:+.6f}  z={z:+.2f}  p={p:.4f}   "
          f"{'SIGNIFICATIVA' if p < 0.05 else 'nao significativa'}")

print("\n" + "=" * 84)
print("PODER ESTATISTICO: quanta heterogeneidade seria detectavel com n=64.000?")
print("=" * 84)
ctrl = df.loc[df["segment"] == "No E-Mail", "conversion"]
p0 = ctrl.mean()
n_por_grupo = 21387
se_ate = np.sqrt(2 * p0 * (1 - p0) / n_por_grupo)
print(f"  taxa de conversao no controle : {p0:.5f}")
print(f"  erro padrao aproximado do ATE : {se_ate:.6f}")
print(f"  diferenca minima detectavel entre dois subgrupos (80% poder, alfa=5%):")
print(f"    ~{2.8 * se_ate * np.sqrt(2):.6f}  (comparando dois subgrupos de metade do tamanho)")
print(f"\n  ATE global observado          : 0.006805")
print(f"  -> so heterogeneidade de magnitude comparavel ao PROPRIO ATE seria detectavel.")
