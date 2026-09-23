"""Monta o arquivo de entrada com `visit` como desfecho, mesma estrutura do de conversion."""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

with urllib.request.urlopen(URL, timeout=180) as r:
    df = pd.read_csv(io.BytesIO(r.read()))
print(f"base baixada: {df.shape}")

X = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
X = pd.concat([X,
               pd.get_dummies(df["zip_code"], prefix="zip", dtype=float),
               pd.get_dummies(df["channel"], prefix="channel", dtype=float)], axis=1)
cod = df["segment"].map({"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2})
T1 = (cod == 1).astype(float)
T2 = (cod == 2).astype(float)

for desfecho in ["visit", "conversion"]:
    Y = df[desfecho].astype(float)
    design = pd.concat([X, Y.rename("y"), T1.rename("T1"), T2.rename("T2")], axis=1)
    arq = os.path.join(RES, f"hillstrom_input_{desfecho}.txt")
    design.to_csv(arq, sep=" ", header=False, index=False)
    print(f"\n{desfecho}: {design.shape[0]:,} x {design.shape[1]}  -> {os.path.basename(arq)}")
    print(f"  taxa geral = {Y.mean()*100:.3f}%")
    for nome, k in [("controle", 0), ("mens", 1), ("womens", 2)]:
        print(f"    {nome:<10} {Y[cod==k].mean()*100:>7.3f}%  (n={int((cod==k).sum()):,})")
    print(f"  ATE mens   = {(Y[cod==1].mean()-Y[cod==0].mean())*100:+.3f} pp")
    print(f"  ATE womens = {(Y[cod==2].mean()-Y[cod==0].mean())*100:+.3f} pp")

print("\nindices: outcome=11  treatment=[12,13]  (identico ao de conversion)")
