"""
S-learner e T-learner na Hillstrom, exploratorio.
Mesmo protocolo dos demais: cross-fitting de 5 dobras, mesmo arquivo de entrada,
predicoes fora-da-amostra para todas as 64.000 linhas.

Aprendiz base: RandomForestClassifier com 300 arvores e min_samples_leaf=100,
espelhando os hiperparametros das florestas de uplift ja rodadas.
"""
import io
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from causalml.inference.meta import BaseSClassifier, BaseTClassifier

SAIDA = (r"C:\Users\m2292\AppData\Local\Temp\claude"
         r"\C--Users-m2292-Documents-tese-mba\f728b89b-8274-4692-8ac2-0528394a10c3"
         r"\scratchpad")

r = subprocess.run(["wsl", "-e", "bash", "-c",
                    "cat ~/lbcf/Data/hillstrom/hillstrom_input.txt"],
                   capture_output=True, text=True)
d = pd.read_csv(io.StringIO(r.stdout), sep=r"\s+", header=None)
X = d.iloc[:, 0:11].values.astype(float)
y = d[11].values.astype(int)
tm = d[12].values.astype(int)
tw = d[13].values.astype(int)
trat = np.where(tm == 1, "1", np.where(tw == 1, "2", "0"))

print(f"dados: {X.shape[0]:,} x {X.shape[1]} | conversao={y.mean():.5f}")
print(f"grupos: controle={int((trat=='0').sum()):,} "
      f"mens={int((trat=='1').sum()):,} womens={int((trat=='2').sum()):,}")
print()


def base():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=100,
                                  n_jobs=-1, random_state=42)


estratos = pd.Series(trat) + "_" + pd.Series(y).astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, Klass in [("S-learner", BaseSClassifier), ("T-learner", BaseTClassifier)]:
    print(f"=== {nome} ===")
    pred = np.full((len(X), 2), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, estratos), 1):
        m = Klass(learner=base(), control_name="0")
        m.fit(X=X[tr], treatment=trat[tr], y=y[tr])
        p = m.predict(X[te])
        # ordena colunas conforme t_groups ('1' = mens, '2' = womens)
        grupos = [str(g) for g in m.t_groups]
        i_m, i_w = grupos.index("1"), grupos.index("2")
        pred[te, 0] = p[:, i_m]
        pred[te, 1] = p[:, i_w]
        print(f"  dobra {k}/5 ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(pred, columns=["mens", "womens"])
    arq = nome.replace("-", "")
    df.to_csv(f"{SAIDA}\\baseline_{arq}.csv", index=False, header=False)
    print(f"  mens  : media={df['mens'].mean():.6f} dp={df['mens'].std():.6f} "
          f"min={df['mens'].min():.6f} max={df['mens'].max():.6f}")
    print(f"  womens: media={df['womens'].mean():.6f} dp={df['womens'].std():.6f} "
          f"min={df['womens'].min():.6f} max={df['womens'].max():.6f}")
    print(f"  salvo: baseline_{arq}.csv")
    print()
