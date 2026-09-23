"""Curvas Qini para o desfecho visit, seis modelos, dois bracos."""
import contextlib
import io as _io
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalml.metrics import get_qini

RES = r"C:\Users\m2292\Documents\tese-mba\resultados"
CV = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\3b21be59-26c0-42ea-a9fc-3082fb63ab50"
      r"\scratchpad\mbcf_cv_visit")
DEST = r"C:\Users\m2292\Documents\tese-mba"

SURF, SERIE, MUTED = "#fcfcfb", "#2a78d6", "#898781"
GRID, AXIS, INK, INK2 = "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
_b = _io.StringIO()


def q(fn, *a, **k):
    with contextlib.redirect_stderr(_b), contextlib.redirect_stdout(_b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)


d = pd.read_csv(os.path.join(RES, "hillstrom_input_visit.txt"), sep=r"\s+", header=None)
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)
n = len(Y)

mods = {}
mods["UDCF default"] = pd.read_csv(os.path.join(RES, "visit_udcf_default"), header=None,
                                   sep=r",\s*", engine="python").values.astype(float)
mods["UDCF ip=0"] = pd.read_csv(os.path.join(RES, "visit_udcf_ip0"), header=None,
                                sep=r",\s*", engine="python").values.astype(float)
idx = np.load(os.path.join(CV, "idx_teste.npy"), allow_pickle=True)
M = np.full((n, 2), np.nan)
for j, arm in enumerate(["mens", "womens"]):
    for f in range(5):
        M[np.asarray(idx[f], dtype=int), j] = pd.read_csv(
            os.path.join(CV, f"pred_{arm}_{f+1}"), header=None)[0].values
mods["MBCF"] = M
for arq, rot in [("visit_Chi.csv", "Chi"), ("visit_ED.csv", "ED"), ("visit_CTS.csv", "CTS")]:
    mods[rot] = pd.read_csv(os.path.join(RES, arq), header=None).values.astype(float)

ORDEM = ["UDCF default", "UDCF ip=0", "MBCF", "Chi", "ED", "CTS"]
ARMS = [("mens", Tm, "Mens E-Mail"), ("womens", Tw, "Womens E-Mail")]

fig, axes = plt.subplots(2, 6, figsize=(17, 6.4), sharex=True, facecolor=SURF)
for i, (arm, T, rot_arm) in enumerate(ARMS):
    sel = np.where(ctrl | (T == 1))[0]
    y, w = Y[sel], T[sel]
    for jx, nome in enumerate(ORDEM):
        ax = axes[i, jx]
        pred = mods[nome][sel, i]
        qq = q(get_qini, pd.DataFrame({"y": y, "w": w, "m": pred}),
               outcome_col="y", treatment_col="w")["m"].values
        x = np.linspace(0, 1, len(qq))
        ax.plot(x, qq[-1] * x, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.plot(x, qq, color=SERIE, lw=2.0, zorder=3)
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(1.0)
        ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
        if i == 0:
            ax.set_title(nome, fontsize=10, color=INK, pad=6)
        if jx == 0:
            ax.set_ylabel(f"{rot_arm}\nvisitas incrementais", fontsize=9, color=INK2)
    lims = [axes[i, k].get_ylim() for k in range(6)]
    lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
    for k in range(6):
        axes[i, k].set_ylim(lo, hi)
        if k > 0:
            axes[i, k].set_yticklabels([])
for k in range(6):
    axes[1, k].set_xlabel("fração da base tratada", fontsize=8.5, color=INK2)

from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=SERIE, lw=2.0, label="modelo"),
                    Line2D([], [], color=MUTED, lw=1.4, ls=(0, (5, 4)), label="aleatório")],
           loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False,
           fontsize=9, labelcolor=INK2, ncol=2)
fig.suptitle("Curvas Qini — desfecho visit, predições fora-da-amostra",
             fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(DEST, "curvas_qini_visit.png"), dpi=200, facecolor=SURF)
print("gravado: curvas_qini_visit.png")
