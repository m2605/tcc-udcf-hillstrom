"""
Prepara os arquivos de validacao cruzada para o MBCF.
Mesmas dobras usadas nas baselines do CausalML (StratifiedKFold, seed 42,
estratificado por tratamento x desfecho), para a comparacao ser pareada.

Para cada dobra f e cada braco a:
  train_{a}_{f}.txt : linhas de treino que sao controle OU do braco a
  test_{f}.txt      : TODAS as linhas retidas da dobra f (qualquer braco)
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
OUT = r"C:\Users\m2292\AppData\Local\Temp\claude\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50\scratchpad\mbcf_cv"
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(RES, "hillstrom_input.txt"), sep=r"\s+", header=None)
X = d.iloc[:, 0:11]
y = d[11]
Tm = d[12].astype(int)
Tw = d[13].astype(int)
ctrl = (Tm == 0) & (Tw == 0)
trat = np.where(Tm == 1, "1", np.where(Tw == 1, "2", "0"))

estratos = pd.Series(trat) + "_" + y.astype(int).astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

idx_teste = {}
for f, (tr, te) in enumerate(skf.split(X.values, estratos), 1):
    # arquivo de teste: TODAS as linhas retidas, qualquer braco
    teste = pd.concat([X.iloc[te], y.iloc[te].rename("y"), Tm.iloc[te].rename("w")], axis=1)
    teste.to_csv(f"{OUT}\\test_{f}.txt", sep=" ", header=False, index=False)
    idx_teste[f] = te

    for arm, T in [("mens", Tm), ("womens", Tw)]:
        # treino: linhas de treino que sao controle ou do braco
        sel = np.array([i for i in tr if ctrl.iloc[i] or T.iloc[i] == 1])
        tre = pd.concat([X.iloc[sel], y.iloc[sel].rename("y"), T.iloc[sel].rename("w")], axis=1)
        tre.to_csv(f"{OUT}\\train_{arm}_{f}.txt", sep=" ", header=False, index=False)
        if f == 1:
            print(f"  dobra 1, {arm}: treino={len(sel):,} "
                  f"(tratados={int(T.iloc[sel].sum()):,}) | teste={len(te):,}")

np.save(f"{OUT}\\idx_teste.npy", np.array([idx_teste[f] for f in range(1, 6)], dtype=object),
        allow_pickle=True)
print(f"\narquivos gravados em {OUT}")
print(f"  5 arquivos de teste + 10 de treino")
print(f"  indices de teste salvos em idx_teste.npy")
