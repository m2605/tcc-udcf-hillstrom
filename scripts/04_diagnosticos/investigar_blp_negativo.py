"""
A floresta DEGENERADA (Etapa 1) deu BLP = -94, p < 0.0001 — altissimamente
"significativo". Mas ela nao tem splits: nao pode ter captado nada.

Hipotese: artefato mecanico do predict_oob. A predicao de i e a media de theta
sobre as arvores que EXCLUIRAM i. Se i contribui positivamente para o efeito
global, as subamostras sem i tem efeito ligeiramente menor -> predicao menor.
Isso gera correlacao NEGATIVA entre a predicao e a contribuicao do proprio i,
sem nenhum sinal real.

Teste: a predicao default depende do desfecho do PROPRIO individuo?
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\m2292\Documents\tese-mba\hillstrom_cate_resultados.csv")
d = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
d["trat"] = (d["segment"] == "Mens E-Mail").astype(int)

print("=" * 76)
print("A predicao da floresta DEGENERADA depende do desfecho do proprio individuo?")
print("(se depende, e vazamento mecanico do OOB, nao sinal)")
print("=" * 76)
print(f"\n{'grupo':<34}{'n':>8}{'cate_default medio':>22}")
print("-" * 76)
for t in [1, 0]:
    for c in [1, 0]:
        s = d[(d["trat"] == t) & (d["conversion"] == c)]
        rot = f"{'tratado' if t else 'controle'}, conversao={c}"
        print(f"{rot:<34}{len(s):>8,}{s['cate_mens_default'].mean():>22.8f}")

print("\nContraste decisivo:")
tc1 = d.loc[(d["trat"] == 1) & (d["conversion"] == 1), "cate_mens_default"]
tc0 = d.loc[(d["trat"] == 1) & (d["conversion"] == 0), "cate_mens_default"]
dif = tc1.mean() - tc0.mean()
se = np.sqrt(tc1.var(ddof=1) / len(tc1) + tc0.var(ddof=1) / len(tc0))
print(f"  tratados: conversao=1 menos conversao=0 = {dif:+.8f}")
print(f"  se = {se:.8f}   z = {dif/se:+.2f}   p = {2*(1-stats.norm.cdf(abs(dif/se))):.2e}")

cc1 = d.loc[(d["trat"] == 0) & (d["conversion"] == 1), "cate_mens_default"]
cc0 = d.loc[(d["trat"] == 0) & (d["conversion"] == 0), "cate_mens_default"]
dif2 = cc1.mean() - cc0.mean()
se2 = np.sqrt(cc1.var(ddof=1) / len(cc1) + cc0.var(ddof=1) / len(cc0))
print(f"  controle: conversao=1 menos conversao=0 = {dif2:+.8f}")
print(f"  se = {se2:.8f}   z = {dif2/se2:+.2f}   p = {2*(1-stats.norm.cdf(abs(dif2/se2))):.2e}")

print("\n  Esperado sob vazamento do OOB:")
print("    tratado convertido   -> puxa o efeito PARA CIMA -> arvores sem ele dao MENOS")
print("    controle convertido  -> puxa o efeito PARA BAIXO -> arvores sem ele dao MAIS")
print("    ou seja: sinais OPOSTOS nos dois contrastes.")
obs = "SIM" if np.sign(dif) != np.sign(dif2) else "NAO"
print(f"    sinais opostos observados? {obs}")

print("\n" + "=" * 76)
print("Mesmo teste na floresta da Etapa 3 (com splits)")
print("=" * 76)
for col in ["cate_mens_ip0"]:
    t1 = d.loc[(d["trat"] == 1) & (d["conversion"] == 1), col]
    t0 = d.loc[(d["trat"] == 1) & (d["conversion"] == 0), col]
    dd = t1.mean() - t0.mean()
    ss = np.sqrt(t1.var(ddof=1) / len(t1) + t0.var(ddof=1) / len(t0))
    print(f"  tratados conv=1 menos conv=0: {dd:+.8f}  z={dd/ss:+.2f}  "
          f"p={2*(1-stats.norm.cdf(abs(dd/ss))):.4f}")
    c1 = d.loc[(d["trat"] == 0) & (d["conversion"] == 1), col]
    c0 = d.loc[(d["trat"] == 0) & (d["conversion"] == 0), col]
    dd2 = c1.mean() - c0.mean()
    ss2 = np.sqrt(c1.var(ddof=1) / len(c1) + c0.var(ddof=1) / len(c0))
    print(f"  controle conv=1 menos conv=0: {dd2:+.8f}  z={dd2/ss2:+.2f}  "
          f"p={2*(1-stats.norm.cdf(abs(dd2/ss2))):.4f}")

print("\n" + "=" * 76)
print("Magnitude relativa: o vazamento explica a dispersao da floresta degenerada?")
print("=" * 76)
print(f"  dp total de cate_mens_default          : {d['cate_mens_default'].std():.8f}")
resid = d["cate_mens_default"] - d.groupby(["trat", "conversion"])["cate_mens_default"].transform("mean")
print(f"  dp apos remover medias por (trat,conv) : {resid.std():.8f}")
r2 = 1 - resid.var() / d["cate_mens_default"].var()
print(f"  fracao da variancia explicada so por (tratamento x desfecho): {r2*100:.1f}%")
