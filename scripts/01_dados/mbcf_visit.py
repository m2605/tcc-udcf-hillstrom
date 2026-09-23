"""Prepara as dobras do MBCF para o desfecho visit."""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
OUT = (r"C:\Users\m2292\AppData\Local\Temp\claude"
       r"\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50"
       r"\scratchpad\mbcf_cv_visit")
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(RES, "hillstrom_input_visit.txt"), sep=r"\s+", header=None)
X = d.iloc[:, 0:11]
y = d[11]
Tm = d[12].astype(int)
Tw = d[13].astype(int)
ctrl = (Tm == 0) & (Tw == 0)
trat = np.where(Tm == 1, "1", np.where(Tw == 1, "2", "0"))

estratos = pd.Series(trat) + "_" + y.astype(int).astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

idx = []
for f, (tr, te) in enumerate(skf.split(X.values, estratos), 1):
    pd.concat([X.iloc[te], y.iloc[te].rename("y"), Tm.iloc[te].rename("w")],
              axis=1).to_csv(f"{OUT}\\test_{f}.txt", sep=" ", header=False, index=False)
    idx.append(te)
    for arm, T in [("mens", Tm), ("womens", Tw)]:
        sel = np.array([i for i in tr if ctrl.iloc[i] or T.iloc[i] == 1])
        pd.concat([X.iloc[sel], y.iloc[sel].rename("y"), T.iloc[sel].rename("w")],
                  axis=1).to_csv(f"{OUT}\\train_{arm}_{f}.txt", sep=" ",
                                 header=False, index=False)

np.save(f"{OUT}\\idx_teste.npy", np.array(idx, dtype=object), allow_pickle=True)
print(f"gravado em {OUT}: 5 testes + 10 treinos")
