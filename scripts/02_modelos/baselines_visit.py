"""Baselines do CausalML + meta-learners, desfecho = visit. Mesmo protocolo."""
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from causalml.inference.tree import UpliftRandomForestClassifier

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
d = pd.read_csv(os.path.join(RES, "hillstrom_input_visit.txt"), sep=r"\s+", header=None)
X = d.iloc[:, 0:11].values.astype(float)
y = d[11].values.astype(int)
tm = d[12].values.astype(int)
tw = d[13].values.astype(int)
trat = np.where(tm == 1, "1", np.where(tw == 1, "2", "0"))
print(f"n={len(X):,}  taxa de visit={y.mean():.5f}", flush=True)

estratos = pd.Series(trat) + "_" + pd.Series(y).astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

PARAMS = dict(n_estimators=300, max_depth=5, min_samples_leaf=100,
              min_samples_treatment=50, n_reg=100, control_name="0",
              n_jobs=1, normalization=True)


def cross_fit(criar, rotulo):
    pred = np.full((len(X), 2), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, estratos), 1):
        m = criar()
        m.fit(X=X[tr], treatment=trat[tr], y=y[tr])
        p = m.predict(X[te])
        if hasattr(m, "classes_"):
            nomes = [str(c) for c in m.classes_ if str(c) != str(m.control_name)]
        else:
            nomes = [str(g) for g in m.t_groups]
        i_m, i_w = nomes.index("1"), nomes.index("2")
        pred[te, 0] = p[:, i_m]
        pred[te, 1] = p[:, i_w]
        print(f"    dobra {k}/5 ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(pred, columns=["mens", "womens"])
    df.to_csv(os.path.join(RES, f"visit_{rotulo}.csv"), index=False, header=False)
    print(f"  media mens={df['mens'].mean():.5f} dp={df['mens'].std():.5f}", flush=True)
    print(f"  media womens={df['womens'].mean():.5f} dp={df['womens'].std():.5f}", flush=True)
    print(f"  salvo: visit_{rotulo}.csv\n", flush=True)


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=100,
                                  n_jobs=-1, random_state=42)


print("=== S-learner ===", flush=True)
cross_fit(lambda: BaseSClassifier(learner=rf(), control_name="0"), "Slearner")
print("=== T-learner ===", flush=True)
cross_fit(lambda: BaseTClassifier(learner=rf(), control_name="0"), "Tlearner")

for crit in ["Chi", "ED", "CTS"]:
    print(f"=== {crit} ===", flush=True)
    cross_fit(lambda c=crit: UpliftRandomForestClassifier(evaluationFunction=c, **PARAMS), crit)

print("CONCLUIDO", flush=True)
