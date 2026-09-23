"""
Baselines Chi / ED / CTS (CausalML) na Hillstrom, com cross-fitting de 5 dobras
para produzir predicoes fora-da-amostra para todas as 64.000 linhas — o analogo
direto do predict_oob usado no UDCF.

Le exatamente o mesmo arquivo de entrada que o UDCF usou.
Hiperparametros: os mesmos que os autores usam em Chi_ED_CTS_train_and_predict-RCT.py
"""
import io
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from causalml.inference.tree import UpliftRandomForestClassifier

SO_TESTE = "--teste" in sys.argv
SAIDA = (r"C:\Users\m2292\AppData\Local\Temp\claude"
         r"\C--Users-m2292-Documents-tese-mba\f728b89b-8274-4692-8ac2-0528394a10c3"
         r"\scratchpad")

# ---- mesmos dados que o UDCF leu ----
r = subprocess.run(["wsl", "-e", "bash", "-c",
                    "cat ~/lbcf/Data/hillstrom/hillstrom_input.txt"],
                   capture_output=True, text=True)
d = pd.read_csv(io.StringIO(r.stdout), sep=r"\s+", header=None)

FEATURES = ["recency", "history", "mens", "womens", "newbie",
            "zip_Rural", "zip_Surburban", "zip_Urban",
            "channel_Multichannel", "channel_Phone", "channel_Web"]
X = d.iloc[:, 0:11].values.astype(float)
y = d[11].values.astype(int)
t_mens = d[12].values.astype(int)
t_womens = d[13].values.astype(int)

# tratamento como string: '0' controle, '1' mens, '2' womens
trat = np.where(t_mens == 1, "1", np.where(t_womens == 1, "2", "0"))

print(f"dados: {X.shape[0]:,} linhas x {X.shape[1]} features")
print(f"grupos: controle={int((trat=='0').sum()):,} "
      f"mens={int((trat=='1').sum()):,} womens={int((trat=='2').sum()):,}")
print(f"taxa de conversao geral: {y.mean():.5f}")
print()

# hiperparametros identicos aos usados pelos autores nas baselines
PARAMS = dict(n_estimators=300, max_depth=5, min_samples_leaf=100,
              min_samples_treatment=50, n_reg=100, control_name="0",
              n_jobs=1, normalization=True)


def ajusta_e_preve(criterio, Xtr, ttr, ytr, Xte):
    m = UpliftRandomForestClassifier(evaluationFunction=criterio, **PARAMS)
    m.fit(X=Xtr, treatment=ttr, y=ytr)
    p = m.predict(Xte)
    # No causalml 0.17, classes_ inclui o controle mas predict() devolve apenas
    # as colunas dos bracos de tratamento, na mesma ordem.
    nomes = [str(c) for c in m.classes_ if str(c) != str(m.control_name)]
    assert len(nomes) == p.shape[1], f"{len(nomes)} nomes vs {p.shape[1]} colunas"
    return pd.DataFrame(p, columns=nomes)


if SO_TESTE:
    print("=== TESTE DE VELOCIDADE: um ajuste em 80% dos dados ===")
    n = int(len(X) * 0.8)
    idx = np.random.default_rng(0).permutation(len(X))
    tr, te = idx[:n], idx[n:]
    t0 = time.time()
    out = ajusta_e_preve("Chi", X[tr], trat[tr], y[tr], X[te])
    dt = time.time() - t0
    print(f"  um ajuste Chi: {dt:.1f}s")
    print(f"  colunas retornadas: {list(out.columns)}")
    print(f"  estimativa para 15 ajustes (5 dobras x 3 criterios): {dt*15/60:.1f} min")
    sys.exit(0)

# ---- cross-fitting ----
estratos = pd.Series(trat) + "_" + pd.Series(y).astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for criterio in ["Chi", "ED", "CTS"]:
    print(f"=== {criterio} ===")
    pred = np.full((len(X), 2), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, estratos), 1):
        out = ajusta_e_preve(criterio, X[tr], trat[tr], y[tr], X[te])
        # ordena as colunas para ficar sempre [mens, womens]
        cols = list(out.columns)
        col_m = [c for c in cols if c.endswith("1")][0]
        col_w = [c for c in cols if c.endswith("2")][0]
        pred[te, 0] = out[col_m].values
        pred[te, 1] = out[col_w].values
        print(f"  dobra {k}/5 concluida ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(pred, columns=["mens", "womens"])
    df.to_csv(f"{SAIDA}\\baseline_{criterio}.csv", index=False, header=False)
    print(f"  media mens={df['mens'].mean():.6f} dp={df['mens'].std():.6f}")
    print(f"  media womens={df['womens'].mean():.6f} dp={df['womens'].std():.6f}")
    print(f"  salvo: baseline_{criterio}.csv")
    print()
