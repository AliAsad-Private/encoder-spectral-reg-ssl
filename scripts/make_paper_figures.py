"""Paper figures from the saved results, logs and checkpoints (runs on CPU)."""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("RES", "/content/drive/MyDrive/dim-ssl-results")
LOGS = os.environ.get("LOGS", "/content/drive/MyDrive/dim-ssl-logs")
CKPT = os.environ.get("CKPT", "/content/drive/MyDrive/dim-ssl-checkpoints")
OUT = os.environ.get("OUT", "/content/drive/MyDrive/dim-ssl-figures")
SKIP_SPECTRUM = os.environ.get("SKIP_SPECTRUM") == "1"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})
C_BASE, C_REG = "#1f4e79", "#b03a2e"

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print("saved", name)

# ---------------- load evaluation results ----------------
R = {}
for f in glob.glob(f"{RES}/*.json"):
    d = json.load(open(f))
    R[os.path.basename(f)[:-5]] = d
# seed 42 (original thesis run, evaluated with the same script)
R.setdefault("baseline_s42_ep399", {"clean_acc": .4849, "mean_corruption_acc": .3901,
             "category_means": {"noise": .3280, "blur": .3972, "weather": .3918, "digital": .4281}})
R.setdefault("lambda05_s42_ep399", {"clean_acc": .4205, "mean_corruption_acc": .3467,
             "category_means": {"noise": .2943, "blur": .3549, "weather": .3475, "digital": .3769}})

base = sorted([(k, v) for k, v in R.items() if k.startswith("baseline")])
reg = sorted([(k, v) for k, v in R.items() if k.startswith("lambda05") and k.endswith("ep399")])
bx = np.array([v["clean_acc"] * 100 for _, v in base])
by = np.array([v["mean_corruption_acc"] * 100 for _, v in base])
rx = np.array([v["clean_acc"] * 100 for _, v in reg])
ry = np.array([v["mean_corruption_acc"] * 100 for _, v in reg])
slope, icpt = np.polyfit(bx, by, 1)
er = ry - (slope * rx + icpt)
print(f"baseline trend: corrupt = {slope:.4f} * clean + {icpt:.3f}  (n={len(bx)})")
print("effective robustness per seed:", np.round(er, 2), "mean %.2f" % er.mean())

# ---------------- Figure: effective robustness ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ep = [int(re.search(r"ep(\d+)", k).group(1)) + 1 for k, _ in base]
sc = ax.scatter(bx, by, c=ep, cmap="Blues", vmin=0, vmax=420, edgecolor=C_BASE,
                linewidth=0.6, s=26, zorder=3, label="Baseline checkpoints")
xs = np.linspace(min(bx.min(), rx.min()) - 0.8, bx.max() + 0.8, 50)
ax.plot(xs, slope * xs + icpt, color=C_BASE, lw=1, ls="--", zorder=2, label="Baseline trend")
ax.scatter(rx, ry, marker="^", color=C_REG, s=34, zorder=4, label=r"$\lambda=0.5$, epoch 400")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Training epoch")
ax.set_xlabel("Clean accuracy (%)")
ax.set_ylabel("Mean corrupted accuracy (%)")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_effective_robustness")

# ---------------- Figure: family-level effective robustness ----------------
fams = ["noise", "blur", "weather", "digital"]
fer = []
for f_ in fams:
    y = np.array([v["category_means"][f_] * 100 for _, v in base])
    s_, i_ = np.polyfit(bx, y, 1)
    fer.append([v["category_means"][f_] * 100 - (s_ * v["clean_acc"] * 100 + i_) for _, v in reg])
fer = np.array(fer)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
xpos = np.arange(len(fams))
ax.bar(xpos, fer.mean(1), width=0.55, color="#bfbfbf", edgecolor="black", linewidth=0.6)
for j in range(fer.shape[1]):
    ax.scatter(xpos + (j - 1) * 0.1, fer[:, j], s=10, color="black", zorder=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xpos, [f.capitalize() for f in fams])
ax.set_ylabel("Effective robustness (points)")
save(fig, "fig_family_effective_robustness")
print("family ER:", dict(zip(fams, np.round(fer.mean(1), 2))))

# ---------------- Figure: training curves from logs ----------------
def parse_log(path):
    t = open(path, errors="ignore").read()
    eps = [int(x) + 1 for x in re.findall(r"Evaluation at epoch (\d+)", t)]
    acc = [float(x) * 100 for x in re.findall(r"best=([\d.]+)", t)]
    rk = [float(x) for x in re.findall(r"Effective rank \(encoder\): ([\d.]+)", t)]
    n = min(len(eps), len(acc), len(rk))
    return np.array(eps[:n]), np.array(acc[:n]), np.array(rk[:n])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
for name, col, lab in [("baseline", C_BASE, "Baseline"), ("lambda05", C_REG, r"$\lambda=0.5$")]:
    for i, f in enumerate(sorted(glob.glob(f"{LOGS}/{name}_s*.log"))):
        e, a, r = parse_log(f)
        kw = dict(color=col, lw=1, marker="o", ms=2.5, alpha=0.85, label=lab if i == 0 else None)
        a1.plot(e, r, **kw)
        a2.plot(e, a, **kw)
a1.set_xlabel("Epoch"); a1.set_ylabel("Effective rank (of 512)")
a2.set_xlabel("Epoch"); a2.set_ylabel("Linear-probe accuracy (%)")
a1.legend(frameon=False)
a1.text(-0.2, 1.02, "(a)", transform=a1.transAxes)
a2.text(-0.2, 1.02, "(b)", transform=a2.transAxes)
fig.tight_layout()
save(fig, "fig_training_curves")

# ---------------- Figure: lambda sweep ----------------
lam = np.array([0, 0.1, 0.25, 0.5, 1.0])
rank = np.array([121.6, 210.0, 302.1, 363.5, 411.8])
clean = np.array([48.49, 48.21, 45.49, 42.05, 34.09])
ret = np.array([80.46, 80.23, 81.83, 82.45, 81.08])
def ms(pfx):
    v = [x["mean_corruption_acc"] / x["clean_acc"] * 100 for k, x in R.items() if k.startswith(pfx) and k.endswith("ep399")]
    return np.mean(v), np.std(v, ddof=1), len(v)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
a1.plot(lam, clean, "o-", color="black", lw=1, ms=3.5)
a1.set_xlabel(r"$\lambda$"); a1.set_ylabel("Clean accuracy (%)")
a1b = a1.twinx(); a1b.spines["right"].set_visible(True)
a1b.plot(lam, rank, "s--", color="gray", lw=1, ms=3.5)
a1b.set_ylabel("Effective rank", color="gray"); a1b.tick_params(axis="y", colors="gray")
a2.plot(lam, ret, "o-", color="black", lw=1, ms=3.5, label="Seed 42")
for l_, pfx in [(0, "baseline"), (0.5, "lambda05")]:
    m, s, n = ms(pfx)
    a2.errorbar(l_ + 0.03, m, yerr=s, fmt="D", color=C_REG, ms=3.5, capsize=2,
                label=f"Mean $\\pm$ s.d., {n} seeds" if l_ == 0 else None)
a2.set_xlabel(r"$\lambda$"); a2.set_ylabel("Retention (%)")
a2.legend(frameon=False, loc="lower right")
a1.text(-0.2, 1.02, "(a)", transform=a1.transAxes)
a2.text(-0.2, 1.02, "(b)", transform=a2.transAxes)
fig.tight_layout()
save(fig, "fig_lambda_sweep")

# ---------------- Figure: eigenvalue spectra (needs checkpoints) ----------------
if not SKIP_SPECTRUM:
    import torch
    sys.path.insert(0, os.getcwd())
    from src.models import SSLModel
    from src.data import get_eval_dataloaders
    from src.evaluation import compute_spectral_diagnostics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_eval_dataloaders("cifar100", "./data", batch_size=500, num_workers=2)
    spectra = {}
    for lab, path in [("Baseline", f"{CKPT}/simclr_baseline_s43/epoch_399.pth"),
                      (r"$\lambda=0.5$", f"{CKPT}/simclr_dimreg_encoder_0.5_s43/epoch_399.pth")]:
        ck = torch.load(path, map_location=dev, weights_only=False)
        c = ck["config"]
        m = SSLModel(backbone=c["backbone"], proj_hidden_dim=c["proj_hidden_dim"],
                     proj_out_dim=c["proj_out_dim"], proj_layers=c["proj_layers"]).to(dev)
        m.load_state_dict(ck["model_state_dict"])
        d = compute_spectral_diagnostics(m, test_loader, device=dev)
        spectra[lab] = d["eigenvalues"]
        print(f"{lab}: effective rank {d['effective_rank']:.1f}")
    np.savez(f"{OUT}/spectra_s43.npz", **{k.replace('$', '').replace('\\', ''): v for k, v in spectra.items()})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for (lab, ev), col in zip(spectra.items(), [C_BASE, C_REG]):
        p = ev / ev.sum()
        a1.semilogy(np.arange(1, len(p) + 1), p, color=col, lw=1.2, label=lab)
        a2.plot(np.arange(1, len(p) + 1), np.cumsum(p), color=col, lw=1.2, label=lab)
    a1.set_xlabel("Eigenvalue index"); a1.set_ylabel("Normalized eigenvalue")
    a2.set_xlabel("Number of directions"); a2.set_ylabel("Cumulative variance")
    a2.axhline(0.9, color="gray", lw=0.6, ls=":")
    a1.legend(frameon=False)
    a1.text(-0.2, 1.02, "(a)", transform=a1.transAxes)
    a2.text(-0.2, 1.02, "(b)", transform=a2.transAxes)
    fig.tight_layout()
    save(fig, "fig_spectrum")
print("done ->", OUT)
