# fig_alpha_scope.py -- generates fig_alpha_scope.pdf (Figure 1 of the paper).
# Everything here is computed from the released metadata and Eq. (3) alone:
# no model output enters this figure.
#   Panel (a): cov_5(alpha) = 5th pct. of Beta(n+1-l, l), l = floor((n+1)alpha),
#              at each stratum's mean de-clustered calibration count (Table 6).
#   Panel (b): Pr(degenerate)(alpha) = Pr(n_cal < ceil(1/alpha) - 1), estimated
#              over patient-level draws under the pre-registered allocations
#              (HIBA: calibration = 70% of patients, unstratified; PAD: training
#              = 40% stratified by phototype, calibration = 60% of the
#              remainder, unstratified), de-clustered per stratum.
# Regenerate with the final audit seed before submission:
#   python fig_alpha_scope.py --draws 4000 --seed <final>
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta as Beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--hiba", default="/mnt/user-data/uploads/1787186450141_hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022.csv")
p.add_argument("--pad", default="/mnt/user-data/uploads/1787186450141_metadata.csv")
p.add_argument("--draws", type=int, default=10000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="fig_alpha_scope.pdf")
args = p.parse_args()
rng = np.random.default_rng(args.seed)
EPS = 1e-9

# ---------------- stratum membership from the metadata ----------------
def hiba_membership(path):
    H = pd.read_csv(path)
    m8 = {"melanoma": "MEL", "nevus": "NV", "basal cell carcinoma": "BCC",
          "squamous cell carcinoma": "SCC", "actinic keratosis": "AK",
          "seborrheic keratosis": "BKL", "solar lentigo": "BKL",
          "lichenoid keratosis": "BKL", "dermatofibroma": "DF",
          "vascular lesion": "VASC"}
    H["cls"] = H["diagnosis"].map(m8)
    les = H.groupby("lesion_id").agg(cls=("cls", "first"),
                                     patient=("patient_id", "first"),
                                     fitz=("fitzpatrick_skin_type", "first")).reset_index()
    les["g"] = les.fitz.map({"I": 1, "II": 2, "III": 3, "IV": 4})
    mal = {"MEL", "BCC", "SCC", "AK"}
    mem = {}
    for pid, sub in les[les.g.notna()].groupby("patient"):
        s = set()
        for _, r in sub.iterrows():
            s.add(("M", int(r.g)) if r.cls in mal else ("B", 0))
        mem[pid] = s
    return mem

def pad_membership(path):
    P = pd.read_csv(path)
    pl = P.groupby(["patient_id", "lesion_id"]).agg(
        cls=("diagnostic", "first"), fitz=("fitspatrick", "first")).reset_index()
    mal = {"MEL", "BCC", "SCC", "ACK"}
    mem = {}
    for pid, sub in pl[pl.fitz.notna()].groupby("patient_id"):
        s = set()
        for _, r in sub.iterrows():
            s.add(("M", int(r.fitz)) if r.cls in mal else ("B", 0))
        mem[pid] = s
    pfitz = P.groupby("patient_id")["fitspatrick"].agg(
        lambda s: tuple(s.dropna().unique()))
    grp = {pid: (int(v[0]) if len(v) else 0) for pid, v in pfitz.items()}
    return mem, grp

# ---------------- patient-level draws under the allocations ----------------
def draw_counts_hiba(mem, strata, ndraws):
    pats = np.array(sorted(mem))
    out = {s: np.empty(ndraws, dtype=int) for s in strata}
    k = int(round(0.7 * len(pats)))
    for i in range(ndraws):
        cal = set(rng.choice(pats, k, replace=False))
        for s in strata:
            out[s][i] = sum(1 for q in cal if s in mem[q])
    return out

def draw_counts_pad(mem, grp, strata, ndraws):
    pats = np.array(sorted(grp))
    g = np.array([grp[q] for q in pats])
    out = {s: np.empty(ndraws, dtype=int) for s in strata}
    for i in range(ndraws):
        tr = set()
        for gv in np.unique(g):
            idx = np.where(g == gv)[0]
            tr.update(pats[rng.choice(idx, int(round(0.4 * len(idx))), replace=False)])
        rem = np.array([q for q in pats if q not in tr])
        cal = set(rng.choice(rem, int(round(0.6 * len(rem))), replace=False))
        for s in strata:
            out[s][i] = sum(1 for q in cal if s in mem.get(q, ()))
    return out

hm = hiba_membership(args.hiba)
pm, pg = pad_membership(args.pad)
H_STRATA = [("M", 1), ("M", 2), ("M", 3), ("B", 0)]
P_STRATA = [("M", 1), ("M", 2), ("M", 3), ("M", 4), ("B", 0)]
hc = draw_counts_hiba(hm, H_STRATA, args.draws)
pc = draw_counts_pad(pm, pg, P_STRATA, args.draws)

# ---------------- analytic cov5 and empirical Pr(degenerate) ----------------
def cov5(alpha, n):
    l = int(np.floor((n + 1) * alpha + EPS))
    if l < 1:
        return np.nan            # below alpha_min: q-hat = +inf, nothing certified
    return Beta.ppf(0.05, n + 1 - l, l)

def pr_deg(alpha, counts):
    floor = int(np.ceil(1.0 / alpha - EPS)) - 1
    return 100.0 * np.mean(counts < floor)

ALPHAS = np.arange(0.02, 0.1601, 0.0002)

# curve spec: (label, color, linestyle, mean-n for panel a, counts for panel b)
C = {"I": "#1f77b4", "II": "#e08214", "III": "#2ca02c", "IV": "#d62728", "B": "#7b52ab"}
curves = [
    ("HIBA mal.\\ I \\& III", C["I"],  "-",  round(hc[("M", 1)].mean()), hc[("M", 1)]),
    ("HIBA mal.\\ II",        C["II"], "-",  round(hc[("M", 2)].mean()), hc[("M", 2)]),
    ("HIBA benign",           C["B"],  "-",  round(hc[("B", 0)].mean()), hc[("B", 0)]),
    ("PAD mal.\\ I",          C["I"],  "--", round(pc[("M", 1)].mean()), pc[("M", 1)]),
    ("PAD mal.\\ II",         C["II"], "--", round(pc[("M", 2)].mean()), pc[("M", 2)]),
    ("PAD mal.\\ III",        C["III"],"--", round(pc[("M", 3)].mean()), pc[("M", 3)]),
    ("PAD mal.\\ IV",         C["IV"], "--", round(pc[("M", 4)].mean()), pc[("M", 4)]),
    ("PAD benign",            C["B"],  "--", round(pc[("B", 0)].mean()), pc[("B", 0)]),
]
curves = [(lab.replace("\\&", "&").replace("\\ ", " "), c, ls, n, cnt)
          for lab, c, ls, n, cnt in curves]

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.6,
    "mathtext.fontset": "cm", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

for lab, col, ls, n, _ in curves:
    y = np.array([cov5(a, n) for a in ALPHAS])
    ax1.plot(ALPHAS, y, ls, color=col, lw=1.4, label=lab)
for lab, col, ls, _, cnt in curves:
    y = np.array([pr_deg(a, cnt) for a in ALPHAS])
    ax2.plot(ALPHAS, y, ls, color=col, lw=1.4)

for ax in (ax1, ax2):
    for a, lw in ((0.05, 0.9), (0.10, 0.5), (0.15, 0.5)):
        ax.axvline(a, color="0.75", lw=lw, ls=":", zorder=0)
    ax.set_xlim(0.02, 0.16)
    ax.set_xlabel(r"requested error level $\alpha$")
    ax.set_xticks([0.02, 0.05, 0.10, 0.15])

# caption anchors on the 34-patient HIBA curve (n = 24)
ax1.plot([0.05, 0.10], [cov5(0.05, 24), cov5(0.10, 24)], "o",
         color=C["I"], ms=3.5, zorder=5)
ax1.annotate("0.883", (0.05, cov5(0.05, 24)), xytext=(3, -9),
             textcoords="offset points", fontsize=7.5, color=C["I"])
ax1.annotate("0.817", (0.10, cov5(0.10, 24)), xytext=(5, -12),
             textcoords="offset points", fontsize=7.5, color=C["I"])
ax1.axhline(0.90, color="0.85", lw=0.6, zorder=0)
ax1.set_ylim(0.60, 1.0)
ax1.set_ylabel(r"$\mathrm{cov}_5$ certified when a threshold exists")
ax1.text(0.02, 1.02, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold")
ax2.set_ylim(-2, 102)
ax2.set_ylabel(r"$\Pr(\mathrm{degenerate})$ [%]")
ax2.text(0.02, 1.02, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")

fig.legend(*ax1.get_legend_handles_labels(), loc="lower center",
           ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(args.out, bbox_inches="tight")

# console check against Table 6 / caption anchors
print(f"draws={args.draws} seed={args.seed}")
for lab, _, _, n, cnt in curves:
    print(f"{lab:16s} mean {cnt.mean():6.1f}  Pr(deg) .05/.10/.15 = "
          f"{pr_deg(0.05, cnt):5.2f} / {pr_deg(0.10, cnt):5.2f} / {pr_deg(0.15, cnt):5.2f} %")
print(f"anchors: cov5(0.05,n=24)={cov5(0.05,24):.3f}  cov5(0.10,n=24)={cov5(0.10,24):.3f}")
