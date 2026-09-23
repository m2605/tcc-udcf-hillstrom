"""
O criterio do UDCF (estagio 2) maximiza theta_left_var + theta_right_var, onde
theta_var e a variancia ENTRE os K efeitos de tratamento dentro do filho.
Com K=2, isso e (theta_mens - theta_womens)^2 / 2.

Ou seja: o UDCF procura regioes onde os bracos se DIFERENCIAM entre si, nao onde
um braco isolado varia. A pergunta certa e: `womens` separa os bracos?
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_udcf_cate_onehot.csv")


def efeitos(sub):
    c = sub.loc[sub["segment"] == "No E-Mail", "conversion"]
    out = {}
    for braco, nome in [("Mens E-Mail", "mens"), ("Womens E-Mail", "womens")]:
        t = sub.loc[sub["segment"] == braco, "conversion"]
        d = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        out[nome] = (d, se)
    return out


print("=" * 84)
print("O CRITERIO DISCRIMINATIVO: |theta_mens - theta_womens| por subgrupo")
print("=" * 84)
print(f"{'subgrupo':<26}{'th_mens':>11}{'th_womens':>11}{'|dif|':>11}{'p(dif)':>9}")
print("-" * 84)

candidatos = [("womens", 0), ("womens", 1), ("mens", 0), ("mens", 1),
              ("newbie", 0), ("newbie", 1),
              ("zip_code", "Urban"), ("zip_code", "Surburban"), ("zip_code", "Rural"),
              ("channel", "Phone"), ("channel", "Web"), ("channel", "Multichannel")]
for col, val in candidatos:
    e = efeitos(df[df[col] == val])
    dif = e["mens"][0] - e["womens"][0]
    se = np.sqrt(e["mens"][1] ** 2 + e["womens"][1] ** 2)
    p = 2 * (1 - stats.norm.cdf(abs(dif / se)))
    print(f"{f'{col}={val}':<26}{e['mens'][0]:>11.6f}{e['womens'][0]:>11.6f}"
          f"{abs(dif):>11.6f}{p:>9.4f}")

print("\n" + "-" * 84)
print("history e recency por quartil (onde a floresta mais dividiu):")
for var in ["history", "recency"]:
    q4 = pd.qcut(df[var], 4, labels=False, duplicates="drop")
    print(f"\n  {var}:")
    for c in sorted(pd.unique(q4)):
        e = efeitos(df[q4 == c])
        dif = e["mens"][0] - e["womens"][0]
        se = np.sqrt(e["mens"][1] ** 2 + e["womens"][1] ** 2)
        p = 2 * (1 - stats.norm.cdf(abs(dif / se)))
        print(f"    Q{c+1}: th_mens={e['mens'][0]:+.6f}  th_womens={e['womens'][0]:+.6f}"
              f"  |dif|={abs(dif):.6f}  p={p:.4f}")

print("\n" + "=" * 84)
print("A SEPARACAO ENTRE BRACOS DIFERE ENTRE womens=0 e womens=1?")
print("=" * 84)
e0, e1 = efeitos(df[df["womens"] == 0]), efeitos(df[df["womens"] == 1])
d0 = e0["mens"][0] - e0["womens"][0]
d1 = e1["mens"][0] - e1["womens"][0]
se0 = np.sqrt(e0["mens"][1] ** 2 + e0["womens"][1] ** 2)
se1 = np.sqrt(e1["mens"][1] ** 2 + e1["womens"][1] ** 2)
dd = d1 - d0
sedd = np.sqrt(se0 ** 2 + se1 ** 2)
p = 2 * (1 - stats.norm.cdf(abs(dd / sedd)))
print(f"  womens=0: theta_mens - theta_womens = {d0:+.6f}")
print(f"  womens=1: theta_mens - theta_womens = {d1:+.6f}")
print(f"  diferenca das diferencas = {dd:+.6f}  se={sedd:.6f}  p={p:.4f}")
print(f"  {'SIGNIFICATIVA -> womens SEPARA os bracos' if p < 0.05 else 'nao significativa'}")
