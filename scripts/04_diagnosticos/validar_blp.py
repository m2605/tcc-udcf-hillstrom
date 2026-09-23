"""
Valida a implementacao do teste BLP/GATES/placebo em dados SINTETICOS onde a
resposta e conhecida, antes de aplicar aos dados reais.

Caso A: heterogeneidade REAL e predicao perfeita  -> BLP deve dar ~1, p pequeno
Caso B: SEM heterogeneidade, predicao e puro ruido -> BLP deve dar ~0, p grande
Caso C: heterogeneidade real mas predicao embaralhada -> BLP ~0
"""
import numpy as np
from scipy import stats

# ---------- implementacao (identica a que vai para o notebook) ----------


def blp(y, w, pred):
    p = w.mean()
    c = pred - pred.mean()
    X = np.column_stack([np.ones(len(y)), w, (w - p) * c])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    Xe = X * r[:, None]
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ (Xe.T @ Xe) @ XtX_inv
    return b[2], float(np.sqrt(np.diag(V))[2])


def rodar(nome, y, w, pred, esperado):
    coef, se = blp(y, w, pred)
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"\n{nome}")
    print(f"  BLP = {coef:+.4f}  se={se:.4f}  z={z:+.2f}  p={p:.4f}")
    print(f"  esperado: {esperado}")
    # GATES
    import pandas as pd
    q = pd.qcut(pred, 5, labels=False, duplicates="drop")
    ates = []
    for k in sorted(np.unique(q)):
        s = q == k
        t, c = y[s & (w == 1)], y[s & (w == 0)]
        ates.append(t.mean() - c.mean())
    print("  GATES Q1->Q5: " + "  ".join(f"{a:+.5f}" for a in ates))
    print(f"  monotono crescente? {all(ates[i] <= ates[i+1] for i in range(len(ates)-1))}")
    return coef


rng = np.random.default_rng(7)
n = 42000
w = rng.integers(0, 2, n).astype(float)
x = rng.normal(size=n)

# --- Caso A: efeito verdadeiro depende de x, predicao = efeito verdadeiro ---
tau = 0.006 + 0.004 * x           # heterogeneidade real
base = 0.006
pA = base + w * tau
yA = (rng.random(n) < np.clip(pA, 0, 1)).astype(float)
rodar("CASO A — heterogeneidade real, predicao perfeita", yA, w, tau,
      "BLP proximo de 1, p pequeno, GATES crescente")

# --- Caso B: efeito constante, predicao e ruido ---
pB = base + w * 0.006
yB = (rng.random(n) < np.clip(pB, 0, 1)).astype(float)
pred_ruido = rng.normal(0.006, 0.0027, n)
rodar("CASO B — SEM heterogeneidade, predicao e ruido", yB, w, pred_ruido,
      "BLP proximo de 0, p grande, GATES achatado")

# --- Caso C: heterogeneidade real, mas predicao embaralhada ---
rodar("CASO C — heterogeneidade real, predicao embaralhada", yA, w,
      rng.permutation(tau), "BLP proximo de 0")

# --- Caso D: predicao correlacionada mas atenuada (caso realista) ---
pred_atenuada = tau * 0.3 + rng.normal(0, 0.004, n)
rodar("CASO D — predicao com sinal parcial", yA, w, pred_atenuada,
      "BLP > 0 e significativo, mas != 1")

# ---------- placebo ----------
print("\n" + "=" * 66)
print("PLACEBO — 200 permutacoes do tratamento (deve centrar em 0)")
print("=" * 66)
for nome, y_, pred_ in [("Caso A (sinal real)", yA, tau),
                        ("Caso B (sem sinal)", yB, pred_ruido)]:
    obs = blp(y_, w, pred_)[0]
    nulos = np.array([blp(y_, rng.permutation(w), pred_)[0] for _ in range(200)])
    p_emp = (np.abs(nulos) >= abs(obs)).mean()
    print(f"  {nome:<22} obs={obs:+.4f}  nulo dp={nulos.std():.4f}  p_emp={p_emp:.4f}")
