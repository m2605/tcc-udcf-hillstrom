"""
Curvas Qini e TOC, em multiplos pequenos: um painel por modelo.
As curvas sao calculadas pelo causalml (get_qini / get_toc); aqui so se desenha.
"""
import contextlib
import io as _io
import subprocess
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalml.metrics import get_qini
from causalml.metrics.visualize import get_cumgain

SC = (r"C:\Users\m2292\AppData\Local\Temp\claude"
      r"\C--Users-m2292-Documents-tese-mba\f728b89b-8274-4692-8ac2-0528394a10c3"
      r"\scratchpad")
DEST = r"C:\Users\m2292\Documents\tese-mba"

# paleta de referencia (valores documentados, superficie clara)
SURF = "#fcfcfb"
SERIE = "#2a78d6"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"

_buf = _io.StringIO()


def quieto(fn, *a, **k):
    with contextlib.redirect_stderr(_buf), contextlib.redirect_stdout(_buf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)


def ler_wsl(caminho, sep=None):
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"cat {caminho}"],
                       capture_output=True, text=True)
    if sep:
        return pd.read_csv(_io.StringIO(r.stdout), sep=sep, header=None, engine="python")
    return pd.read_csv(_io.StringIO(r.stdout), header=None)


d = ler_wsl("~/lbcf/Data/hillstrom/hillstrom_input.txt", sep=r"\s+")
Y = d[11].values.astype(float)
Tm = d[12].values.astype(float)
Tw = d[13].values.astype(float)
ctrl = (Tm == 0) & (Tw == 0)

preds = {}
for nome, cam in [("UDCF default", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip001"),
                  ("UDCF ip=0", "~/lbcf/Code/Model/LBCF/output/abl_udcf_ip0")]:
    m = ler_wsl(cam)
    m.columns = ["mens", "womens"]
    preds[nome] = m
for arq, rot in [("Chi", "Chi"), ("Tlearner", "T-learner")]:
    m = pd.read_csv(f"{SC}\\baseline_{arq}.csv", header=None)
    m.columns = ["mens", "womens"]
    preds[rot] = m
mb = {a: ler_wsl(f"~/mbcf/out/mbcf_{a}_ip0.01")[0].values.astype(float)
      for a in ["mens", "womens"]}

MODELOS = ["UDCF default", "UDCF ip=0", "MBCF default", "Chi", "T-learner"]
ARMS = [("mens", Tm, "Mens E-Mail"), ("womens", Tw, "Womens E-Mail")]


def estilo(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)


def painel_qini(ax, y, w, pred, titulo):
    df = pd.DataFrame({"y": y, "w": w, "m": pred})
    q = quieto(get_qini, df, outcome_col="y", treatment_col="w")
    curva = q["m"].values
    x = np.linspace(0, 1, len(curva))
    # ranking aleatorio: reta de (0,0) ao valor final da curva
    aleatorio = curva[-1] * x
    ax.plot(x, aleatorio, color=MUTED, linewidth=1.4,
            linestyle=(0, (5, 4)), zorder=2)
    ax.plot(x, curva, color=SERIE, linewidth=2.0, zorder=3)
    estilo(ax)
    ax.set_title(titulo, fontsize=9.5, color=INK, pad=6)


for fig_nome, arms_sub in [("curvas_qini", ARMS)]:
    fig, axes = plt.subplots(2, len(MODELOS), figsize=(15, 6.2),
                             sharex=True, facecolor=SURF)
    for i, (arm, t_arm, rot_arm) in enumerate(arms_sub):
        sel = np.where(ctrl | (t_arm == 1))[0]
        y = Y[sel]
        w = t_arm[sel]
        ymin, ymax = 0, 0
        curvas = []
        for nome in MODELOS:
            p = mb[arm] if nome == "MBCF default" else preds[nome][arm].values[sel]
            curvas.append(p)
        for j, (nome, p) in enumerate(zip(MODELOS, curvas)):
            painel_qini(axes[i, j], y, w, p, nome if i == 0 else "")
            if j == 0:
                axes[i, j].set_ylabel(f"{rot_arm}\nrespondentes incrementais",
                                      fontsize=9, color=INK2)
        # escala comum por linha
        lims = [axes[i, j].get_ylim() for j in range(len(MODELOS))]
        lo = min(l[0] for l in lims)
        hi = max(l[1] for l in lims)
        for j in range(len(MODELOS)):
            axes[i, j].set_ylim(lo, hi)
            if j > 0:
                axes[i, j].set_yticklabels([])
    for j in range(len(MODELOS)):
        axes[1, j].set_xlabel("fração da base tratada", fontsize=8.5, color=INK2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=SERIE, lw=2.0, label="modelo"),
                        Line2D([], [], color=MUTED, lw=1.4,
                               linestyle=(0, (5, 4)), label="aleatório")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995),
               frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
    fig.suptitle("Curvas Qini — Hillstrom, predições fora-da-amostra",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{DEST}\\{fig_nome}.png", dpi=200, facecolor=SURF)
    print(f"gravado: {fig_nome}.png")
