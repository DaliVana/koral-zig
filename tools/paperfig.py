#!/usr/bin/env python3
"""paperfig.py — the test-problem figures of the koral-zig paper.

The reproduction tests assert; this script draws what they asserted on.
Each test that owns a figure dumps its data when the suite is run with
-Dfig-dir (koral/testing/figdata.zig):

    zig build test -Doptimize=ReleaseFast -Dslow-tests -Dfig-dir=dumps/figdata
    python3 tools/paperfig.py dumps/figdata [--out DIR] [--only NAME ...]

One figure per test problem, written as PNG and PDF. A figure whose dump is
missing is skipped (a default, non-slow run produces only the *_fast dumps),
and the script says which. Needs numpy and matplotlib.

Figures (dump file -> figure):
  shocktube     k13_shocktube            Sadowski+ 2013 Fig. 1
  torus         torus_hydro              cf. Sadowski+ 2013 Fig. 2
  radtubes      k13_radtubes_{slow,fast} Sadowski+ 2013 Figs. 3-7
  pulse_thin    k13_pulse_thin_{axis,map} Sadowski+ 2013 Fig. 8
  pulse_thick   k13_pulse_thick[_fast]   Sadowski+ 2013 Fig. 9
  shadow        k13_shadow_{map,cut}     Sadowski+ 2013 Fig. 10
  atmosphere    k13_atmosphere[_short|_fast] Sadowski+ 2013 Fig. 11
  beam          k13_beam_{maps,geodesics} Sadowski+ 2013 Fig. 12
  radbondi      k13_radbondi             Sadowski+ 2013 Fig. 13 setup (not a reproduction)
  waves         k14_waves[_fast_*], k14_waves_ppm  Sadowski+ 2014 Fig. 3
  magtubes      k14_magtube_{2a,5a}      Sadowski+ 2014 Figs. 4-5
  orszag        k14_orszag_{map,cut}     Sadowski+ 2014 Fig. 6
  bondi         bondi_1d, bondi_2d[_fast], bondi_mag[_fast]
  alfven        alfven_wave
  deflection    grrt_deflection          Gold+ 2020 Fig. 1 (imager)
  ipole         grrt_ipole               koral-zig vs ipole on one snapshot;
                                         written by tools/ipolecheck.py compare

Colour: categorical series take the first four slots of the validated
palette in fixed order (blue, orange, aqua, yellow) and always carry a
second encoding (marker or dash) because the last two are light on paper;
magnitudes use one blue ramp, signed residuals blue-grey-red.
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# ---- palette (validated: categorical slots 1-4, light surface) -------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]
DASH = ["-", "--", "-.", ":"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d3"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#e66767", "#a32323"])

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# ---- reading the dumps ------------------------------------------------------
class Block:
    def __init__(self, label):
        self.label = label
        self.meta = {}
        self.rows = []
        self.data = None

    def nums(self):
        """All numbers in the block label, in order."""
        return [float(x) for x in re.findall(FLOAT, self.label)]


class Dump:
    def __init__(self, path):
        self.meta = {}
        self.blocks = []
        cur = None
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    key, _, val = line[1:].partition("=")
                    key, val = key.strip(), val.strip()
                    if key == "block":
                        cur = Block(val)
                        self.blocks.append(cur)
                    elif cur is not None:
                        cur.meta[key] = val
                    else:
                        self.meta[key] = val
                elif line.strip():
                    if cur is None:
                        cur = Block("")
                        self.blocks.append(cur)
                    cur.rows.append([float(x) for x in line.split()])
        for b in self.blocks:
            b.data = np.array(b.rows) if b.rows else np.zeros((0, 0))
            b.rows = None
            # file-level header lines are defaults for every block
            for k, v in self.meta.items():
                b.meta.setdefault(k, v)

    def find(self, pattern):
        return [b for b in self.blocks if re.search(pattern, b.label)]


class Missing(Exception):
    pass


def load(d, *names):
    """First existing dump among `names`."""
    for n in names:
        p = os.path.join(d, n + ".dat")
        if os.path.exists(p):
            return Dump(p)
    raise Missing(" or ".join(n + ".dat" for n in names))


def grid2d(b, shape=None):
    """Rows of (x, y, v...) written y-outer, x-inner -> arrays [ny, nx]."""
    if shape is None:
        ny, nx = (int(v) for v in b.meta["shape"].split())
    else:
        ny, nx = shape
    return [b.data[:, k].reshape(ny, nx) for k in range(b.data.shape[1])]


def guide(ax, x, y0, order, label):
    """A dashed power-law guide line through (x[0], y0)."""
    x = np.asarray(x, dtype=float)
    ax.plot(x, y0 * (x / x[0]) ** (-order), color=INK2, lw=0.8, ls=(0, (4, 3)), label=label)


def orders(n, e):
    return np.log2(e[:-1] / e[1:]) / np.log2(n[1:] / n[:-1])


def sci(v):
    """1e5 -> '10^{5}', 3e3 -> '3\\times10^{3}' (mathtext, no dollars)."""
    m, e = f"{v:.0e}".split("e")
    return (f"{m}\\times" if m != "1" else "") + f"10^{{{int(e)}}}"


def panel_label(ax, text, loc="tl"):
    xy = {"tl": (0.03, 0.95, "left", "top"), "tr": (0.97, 0.95, "right", "top"), "bl": (0.03, 0.05, "left", "bottom"), "c": (0.5, 0.5, "center", "center")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8, color=INK2)


# ---- figures ----------------------------------------------------------------
def fig_shocktube(d):
    D = load(d, "k13_shocktube")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    names = [r"density $D=\rho u^t$", r"velocity $v$", r"pressure $p$"]
    for k, ax in enumerate(axes):
        ex = D.blocks[0].data
        ax.plot(ex[:, 0], ex[:, 4 + k], color=INK, lw=1.0, label="exact")
        for j, b in enumerate(D.blocks):
            th = b.nums()[0]
            ax.plot(b.data[::4, 0], b.data[::4, 1 + k], ls="none", marker=MARK[j], ms=3, mfc="none", mew=0.9, color=C[j], label=rf"$\theta={th:g}$")
        ax.set_xlabel("$x$")
        ax.set_title(names[k], fontsize=9)
        if k == 2:
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle("Relativistic shock tube at $t=50$, 500 cells, linear reconstruction (Sądowski et al. 2013, Fig. 1)", fontsize=10)
    return fig


def fig_torus(d):
    D = load(d, "torus_hydro")
    maps = D.find(r"^map")
    b = maps[-1]
    r, th, rho0, rho = grid2d(b)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    x, z = r * np.sin(th), r * np.cos(th)
    peak = rho0.max()
    # frame the torus: everything above 1e-3 of the peak, with a margin
    body = rho0 > 1e-3 * peak
    xhi = 1.08 * x[body].max()
    zhi = max(1.3 * np.abs(z[body]).max(), 0.3 * xhi)
    ax = axes[0]
    im = ax.pcolormesh(x, z, np.maximum(rho / peak, 1e-6), norm=LogNorm(1e-4, 1.0), cmap=SEQ, shading="gouraud", rasterized=True)
    ax.contour(x, z, rho0 / peak, levels=[1e-3, 1e-2, 1e-1, 0.5], colors=INK, linewidths=0.6)
    ax.set_xlim(0, xhi)
    ax.set_ylim(-zhi, zhi)
    ax.set_xlabel(r"$r\sin\theta$")
    ax.set_ylabel(r"$r\cos\theta$")
    ax.set_title(rf"$\rho/\rho_\mathrm{{max}}$ at $t={b.nums()[-1]:.0f}$ (colour), initial (contours)", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01)
    ax = axes[1]
    mask = rho0 > 1e-3 * peak
    res = np.where(mask, rho / rho0 - 1.0, 0.0)
    lim = np.max(np.abs(res))
    im = ax.pcolormesh(x, z, res, cmap=DIV, vmin=-lim, vmax=lim, shading="gouraud", rasterized=True)
    ax.set_xlim(0, xhi)
    ax.set_ylim(-zhi, zhi)
    ax.set_xlabel(r"$r\sin\theta$")
    ax.set_title(r"$\rho/\rho_0-1$ where $\rho_0>10^{-3}\rho_\mathrm{max}$", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01)
    ax = axes[2]
    conv = D.find("convergence")
    if conv:
        n, e = conv[0].data[:, 0], conv[0].data[:, 1]
        ax.loglog(n, e, marker=MARK[0], color=C[0], label=r"$L_1$ drift of $\rho$")
        guide(ax, n, e[0] * 1.6, 2, r"$N^{-2}$")
        panel_label(ax, "orders " + " ".join(f"{o:.2f}" for o in orders(n, e)), "bl")
        ax.set_xticks(n)
        ax.set_xticklabels([f"{int(v)}" for v in n])
        ax.minorticks_off()
        ax.legend()
    ax.set_xlabel(r"cells per dimension $N$")
    ax.set_title("stationarity error", fontsize=9)
    fig.suptitle(f"Hydro-only equilibrium torus on the MKS2 grid, ${b.label.split()[1].replace('x', chr(92) + 'times')}$ cells (cf. Sądowski et al. 2013, Fig. 2)", fontsize=10)
    return fig


def fig_radtubes(d):
    D = load(d, "k13_radtubes_slow", "k13_radtubes_fast")
    # the best-resolved block of each tube
    best = {}
    for b in D.blocks:
        name = b.label.split()[1]
        if name not in best or b.data.shape[0] > best[name].data.shape[0]:
            best[name] = b
    tubes = [t for t in ["1", "2", "3a", "3b", "4a", "4b"] if t in best]
    cols = [(r"$\rho$", 1), (r"$u^x$", 3), (r"$\widehat{E}$", 4), (r"$\widehat{F}^x$", 5)]
    fig, axes = plt.subplots(len(tubes), 4, figsize=(11.0, 1.75 * len(tubes) + 0.6), sharex=True, constrained_layout=True, squeeze=False)
    for i, t in enumerate(tubes):
        b = best[t]
        s = [float(v) for v in re.findall(FLOAT, b.meta.get("states", ""))]
        for j, (name, col) in enumerate(cols):
            ax = axes[i, j]
            ax.plot(b.data[:, 0], b.data[:, col], color=C[0])
            if len(s) == 8 and col in (1, 3, 4):
                k = {1: 0, 3: 2, 4: 3}[col]
                ax.axhline(s[k], xmax=0.25, color=INK2, lw=0.8, ls=(0, (4, 3)))
                ax.axhline(s[4 + k], xmin=0.75, color=INK2, lw=0.8, ls=(0, (4, 3)))
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"tube {t}\n{b.data.shape[0]} cells", fontsize=8)
            if i == len(tubes) - 1:
                ax.set_xlabel("$x$")
    fig.suptitle("Radiative shock tubes at stationarity (Sądowski et al. 2013, Figs. 3-7); dashed: the asymptotic states", fontsize=10)
    return fig


def fig_pulse_thick(d):
    D = load(d, "k13_pulse_thick", "k13_pulse_thick_fast")
    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    later = [b for b in D.blocks if b.nums()[0] > 0]
    b0 = D.blocks[0]
    ax.plot(b0.data[:, 0], b0.data[:, 1], color=INK2, lw=1.0, label="$t=0$")
    for j, b in enumerate(later):
        t = b.nums()[0]
        ax.plot(b.data[:, 0], b.data[:, 1], color=C[j % 4], ls=DASH[j % 4], label=rf"$t={sci(t)}$")
        ax.plot(b.data[::3, 0], b.data[::3, 2], ls="none", marker=".", ms=3, color=INK)
    ax.plot([], [], ls="none", marker=".", ms=3, color=INK, label="exact diffusion")
    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$E/E_\mathrm{background}$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Optically thick pulse, 101 cells (Sądowski et al. 2013, Fig. 9)", fontsize=10)
    return fig


def fig_pulse_thin(d):
    A = load(d, "k13_pulse_thin_axis")
    M = load(d, "k13_pulse_thin_map")
    n = len(A.blocks)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6.0), constrained_layout=True)
    peaks = [float(v) for v in re.findall(FLOAT, A.meta.get("paper_peaks", ""))]
    paper = {}
    if len(peaks) >= 6:
        paper = {peaks[0]: (peaks[1], peaks[2]), peaks[3]: (peaks[4], peaks[5])}
    for k in range(n):
        t = A.blocks[k].nums()[0]
        x, y, e = grid2d(M.blocks[k])
        ax = axes[0, k]
        im = ax.pcolormesh(x, y, np.maximum(e, 1e-8), norm=LogNorm(1e-7, 1.0), cmap=SEQ, shading="nearest", rasterized=True)
        ph = np.linspace(0, 2 * np.pi, 200)
        ax.plot((5 + t) * np.cos(ph), (5 + t) * np.sin(ph), color=C[1], lw=0.9, ls="--")
        ax.set_aspect("equal")
        ax.set_title(rf"$t={t:g}$", fontsize=10)
        ax.set_xlabel("$x$")
        if k == 0:
            ax.set_ylabel("$y$")
        ax.grid(False)
        ax = axes[1, k]
        a = A.blocks[k].data
        ax.semilogy(a[:, 0], np.maximum(a[:, 1], 1e-9), color=C[0], marker="o", ms=2.5, lw=1.0, label="koral-zig")
        if t in paper:
            ax.plot([paper[t][0], -paper[t][0]], [paper[t][1]] * 2, ls="none", marker="D", ms=6, mfc="none", mew=1.2, color=C[1], label="paper's peak (digitised)")
        if t > 0:
            xs = np.linspace(8, 49, 50)
            pk = a[:, 1].max()
            xp = abs(a[np.argmax(a[:, 1]), 0])
            ax.plot(xs, pk * (xs / xp) ** -2, color=INK2, lw=0.8, ls=(0, (4, 3)), label=r"$\propto x^{-2}$")
        ax.set_ylim(1e-9, 3)
        ax.set_xlabel("$x$")
        if k == 0:
            ax.set_ylabel(r"$E_\mathrm{lab}/E_\mathrm{lab}(0,t{=}0)$")
        ax.legend(loc="upper right" if t == 0 else "upper center", fontsize=7)
    fig.colorbar(im, ax=axes[0, :], pad=0.01, aspect=30, label=r"$E_\mathrm{lab}/E_\mathrm{lab}(0,t{=}0)$")
    fig.suptitle("Optically thin pulse, $51^3$ cells: $z=0$ plane (top; dashed circle $r=w+ct$) and the $x$ axis (bottom) (Sądowski et al. 2013, Fig. 8)", fontsize=10)
    return fig


def fig_shadow(d):
    M = load(d, "k13_shadow_map")
    Cut = load(d, "k13_shadow_cut")
    x, y, e, fx, fy, rho = grid2d(M.blocks[0])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.3), constrained_layout=True, gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    im = ax.pcolormesh(x, y, e, cmap=SEQ, vmin=0, vmax=1.02, shading="nearest", rasterized=True)
    cs = ax.contour(x, y, rho, levels=[1, 100, 500], colors=[C[1]], linewidths=0.9)
    ax.clabel(cs, fmt="%g", fontsize=7)
    sk = (slice(2, None, 5), slice(2, None, 6))
    ax.quiver(x[sk], y[sk], (fx * e)[sk], (fy * e)[sk], color=INK, scale=28, width=0.0022)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(False)
    ax.set_title(r"$E/E_L$ (colour), $\rho$ = 1, 100, 500 (contours), flux $F^i/E_L$ (arrows)", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01)
    ax = axes[1]
    c = Cut.blocks[0].data
    ax.plot(c[:, 0], c[:, 2], color=INK, lw=1.0, label=r"exact $e^{-\tau(y)}$")
    ax.plot(c[:, 0], c[:, 1], ls="none", marker=MARK[0], ms=3.5, mfc="none", mew=0.9, color=C[0], label="koral-zig, $x=2$")
    ax.set_xlabel("$y$")
    ax.set_ylabel("$E/E_L$")
    ax.legend(loc="lower right")
    l1 = np.mean(np.abs(c[:, 1] - c[:, 2]))
    panel_label(ax, f"$L_1$ = {l1:.1e}", "c")
    ax.set_title("transmitted light behind the blob", fontsize=9)
    fig.suptitle("Single-beam shadow at $t=10$, $100\\times50$ cells, M1 closure (Sądowski et al. 2013, Fig. 10)", fontsize=10)
    return fig


def fig_atmosphere(d):
    D = load(d, "k13_atmosphere", "k13_atmosphere_short", "k13_atmosphere_fast")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharex=True, constrained_layout=True)
    tend = D.blocks[0].nums()[-1]
    for j, b in enumerate(D.blocks):
        f = b.nums()[0]
        lab = rf"$L={f:g}\,L_\mathrm{{Edd}}$" if f > 1e-5 else r"$L=10^{-10}\,L_\mathrm{Edd}$"
        a = b.data
        r = a[:, 0] / 1e6
        c, m, ls = C[j % 4], MARK[j % 4], DASH[j % 4]
        axes[0, 0].plot(r, a[:, 2], color=c, ls=ls, lw=1.0)
        axes[0, 0].plot(r, a[:, 1], ls="none", marker=m, ms=3.5, mfc="none", mew=0.9, color=c, label=lab)
        axes[0, 1].plot(r, a[:, 1] / a[:, 2] - 1.0, color=c, ls=ls, marker=m, ms=2.5, label=lab)
        axes[1, 0].plot(r, a[:, 3] / a[:, 4] - 1.0, color=c, ls=ls, marker=m, ms=2.5, label=lab)
        axes[1, 1].plot(r, a[:, 5], color=c, ls=ls, marker=m, ms=2.5, label=lab)
    axes[0, 0].set_ylabel(r"$\rho/\rho_\mathrm{in}$")
    axes[0, 0].set_title("density: numerical (symbols), analytic (lines)", fontsize=9)
    axes[0, 1].set_title(r"density residual $\rho/\rho_\mathrm{analytic}-1$", fontsize=9)
    axes[0, 1].set_yscale("symlog", linthresh=1e-4)
    axes[1, 0].set_title(r"flux residual $F r^2/(F_\mathrm{in} r_\mathrm{in}^2)-1$", fontsize=9)
    axes[1, 1].set_title(r"radial velocity $v^r/c$", fontsize=9)
    axes[1, 1].set_yscale("symlog", linthresh=1e-8)
    for ax in axes[1]:
        ax.set_xlabel(r"$r/10^6\,r_g$")
    axes[0, 0].legend()
    fig.suptitle(rf"Static atmosphere at $t={sci(tend)}\,M$, 40 cells (Sądowski et al. 2013, Fig. 11)", fontsize=10)
    return fig


def fig_beam(d):
    M = load(d, "k13_beam_maps")
    G = load(d, "k13_beam_geodesics")
    models = sorted({b.label.split()[1] for b in M.blocks})
    grids = sorted({b.label.split()[3] for b in M.blocks}, key=lambda g: int(g.split("x")[0]))
    fig, axes = plt.subplots(len(grids), len(models), figsize=(3.6 * len(models), 3.5 * len(grids)), constrained_layout=True, squeeze=False)
    for i, gname in enumerate(grids):
        for j, mname in enumerate(models):
            b = [bb for bb in M.blocks if bb.label.split()[1] == mname and bb.label.split()[3] == gname][0]
            nr, nphi = (int(v) for v in gname.split("x"))
            r, ph, e = grid2d(b, (nphi, nr))
            ax = axes[i, j]
            im = ax.pcolormesh(r * np.cos(ph), r * np.sin(ph), e, cmap=SEQ, vmin=0, vmax=1.05, shading="gouraud", rasterized=True)
            g = [gg for gg in G.blocks if gg.label.split()[1] == mname][0].data
            rmax = r.max()
            for col, ls, lab in ((1, "-", "geodesic of the beam centre"), (2, "--", "geodesics of the beam edges"), (3, "--", None)):
                ok = g[:, col] <= rmax
                ax.plot((g[:, col] * np.cos(g[:, 0]))[ok], (g[:, col] * np.sin(g[:, 0]))[ok], color=C[1], lw=1.0, ls=ls, label=lab)
            ax.set_aspect("equal")
            ax.set_xlim(0, rmax)
            ax.set_ylim(0, rmax)
            ax.grid(False)
            ax.set_xlabel(r"$r\cos\phi$")
            if j == 0:
                ax.set_ylabel(rf"${nr}\times{nphi}$ cells" + "\n" + r"$r\sin\phi$")
            if i == 0:
                rb = b.nums()[3]
                ax.set_title(rf"model {mname}: $r_\mathrm{{beam}}={rb:g}$", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(loc="lower left", fontsize=7)
    fig.colorbar(im, ax=axes, pad=0.01, aspect=40, label=r"$E/E_\mathrm{injected}$")
    fig.suptitle("Light beams near a Schwarzschild black hole (Sądowski et al. 2013, Fig. 12)", fontsize=10)
    return fig


def fig_radbondi(d):
    D = load(d, "k13_radbondi")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)
    for j, b in enumerate(D.blocks):
        a = b.data
        n = int(b.nums()[-2]) if len(b.nums()) >= 2 else a.shape[0]
        lab = f"{a.shape[0]} cells"
        axes[0].loglog(a[:, 0], a[:, 2], color=C[j], ls=DASH[j], label=lab)
        axes[1].loglog(a[:, 0], a[:, 2] / a[:, 3], color=C[j], ls=DASH[j], label=lab)
        pos = a[:, 5] > 0
        axes[2].loglog(a[pos, 0], a[pos, 5], color=C[j], ls=DASH[j], label=lab)
        _ = n
    a = D.blocks[0].data
    axes[0].loglog(a[:, 0], a[:, 3], color=INK, lw=1.0, label="adiabat (initial)")
    axes[1].axhline(1.0, color=INK, lw=1.0)
    lp = float(D.meta.get("paper_luminosity", "nan"))
    axes[2].axhline(lp, color=INK, lw=1.0, label="paper, $r=10^3$ (512 cells)")
    axes[0].set_ylabel(r"$T_\mathrm{gas}$ [K]")
    axes[1].set_ylabel(r"$T_\mathrm{gas}/T_\mathrm{adiabat}$")
    axes[2].set_ylabel(r"$L(r)/L_\mathrm{Edd}$")
    for ax in axes:
        ax.set_xlabel("$r$")
        ax.legend()
    fig.suptitle("Radiative Bondi accretion, model E10T6 at $t=2000$: NOT a reproduction — spurious heating of the Mach-800 flow on coarse grids", fontsize=10)
    return fig


def _wave_panels(D):
    rows = {"tau=0": [], "tau=0.1": [], "tau=10": []}
    for b in D.blocks:
        mode = b.label.split("|")[0].strip()
        tau, kind = mode.split()
        rows[tau].append((kind, b))
    return rows


def fig_waves(d):
    try:
        dumps = [load(d, "k14_waves")]
    except Missing:
        dumps = [load(d, n) for n in ("k14_waves_fast_tau0", "k14_waves_fast_thin", "k14_waves_fast_thick")]
    rows = {"tau=0": [], "tau=0.1": [], "tau=10": []}
    for D in dumps:
        for k, v in _wave_panels(D).items():
            rows[k] += v
    ppm = None
    try:
        ppm = load(d, "k14_waves_ppm")
    except Missing:
        pass
    ncol = 4 if ppm else 3
    fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.5), sharey=True, constrained_layout=True)
    titles = {"tau=0": "no radiation", "tau=0.1": r"optically thin, $\tau=0.1$", "tau=10": r"optically thick, $\tau=10$"}
    names = {"sonic": "sonic ($B=0$)", "fast": "fast", "slow": "slow"}
    for ax, key in zip(axes, rows):
        for j, (kind, b) in enumerate(rows[key]):
            n, e = b.data[:, 0], b.data[:, 1]
            ax.loglog(n, e, color=C[j], marker=MARK[j], ls=DASH[j], label=names.get(kind, kind))
        if rows[key]:
            n = rows[key][0][1].data[:, 0]
            e0 = max(b.data[0, 1] for _, b in rows[key])
            guide(ax, n, 2.5 * e0, 2, r"$N^{-2}$")
            guide(ax, n, 2.5 * e0, 1, None)
            ax.text(n[-1], 2.5 * e0 * (n[-1] / n[0]) ** -1 * 1.15, r"$N^{-1}$", fontsize=7, color=INK2, ha="right", va="bottom")
            ax.set_xticks(n)
            ax.set_xticklabels([f"{int(v)}" for v in n])
            ax.minorticks_off()
        ax.set_title(titles[key], fontsize=9)
        ax.set_xlabel(r"$N_X$")
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"$L_1$ error of $\rho$ after one period")
    if ppm:
        ax = axes[3]
        for j, b in enumerate(ppm.blocks):
            mode, _recon, cfl = [s.strip() for s in b.label.split("|")]
            n, e = b.data[:, 0], b.data[:, 1]
            tau = mode.split()[0].replace("tau=", "")
            ax.loglog(n, e, color=C[j], marker=MARK[j], ls=DASH[j], label=rf"$\tau={tau}$, CFL {cfl.split()[-1]}")
        n = ppm.blocks[0].data[:, 0]
        guide(ax, n, 2.0 * ppm.blocks[0].data[0, 1], 2, r"$N^{-2}$")
        ax.set_xticks(n)
        ax.set_xticklabels([f"{int(v)}" for v in n])
        ax.minorticks_off()
        ax.set_title("production PPM (sonic modes)", fontsize=9)
        ax.set_xlabel(r"$N_X$")
        ax.legend(loc="lower left")
    fig.suptitle("Radiation-modified MHD linear waves, linear reconstruction $\\theta=1.5$ (Sądowski et al. 2014, Fig. 3)", fontsize=10)
    return fig


def fig_magtubes(d):
    T2 = load(d, "k14_magtube_2a")
    T5 = load(d, "k14_magtube_5a")
    # columns: x rho p vx vy vz By Bz ; the paper's panels per tube
    panels = {
        "2a": [(1, r"$\rho$"), (3, "$v_x$"), (5, "$v_z$"), (6, "$B_y$"), (7, "$B_z$")],
        "5a": [(1, r"$\rho$"), (3, "$v_x$"), (4, "$v_y$"), (6, "$B_y$"), (2, "$p$")],
    }
    fig, axes = plt.subplots(5, 2, figsize=(8.6, 9.0), sharex=True, constrained_layout=True)
    for j, (name, D) in enumerate((("2a", T2), ("5a", T5))):
        a = D.blocks[0].data
        for i, (col, lab) in enumerate(panels[name]):
            ax = axes[i, j]
            ax.plot(a[:, 0], a[:, col], color=C[0], ls="none", marker="o", ms=1.6)
            ax.set_ylabel(lab)
        t = re.search(r"at t = (" + FLOAT + ")", D.meta.get("source", ""))
        axes[0, j].set_title(f"tube {name}, $t={float(t.group(1)):g}$, {a.shape[0]} cells" if t else f"tube {name}", fontsize=10)
        axes[4, j].set_xlabel("$x$")
    fig.suptitle("Ryu & Jones magnetic shock tubes, unscaled units (Sądowski et al. 2014, Figs. 4-5)", fontsize=10)
    return fig


def fig_orszag(d):
    M = load(d, "k14_orszag_map")
    Cut = load(d, "k14_orszag_cut")
    x, y, rho, pmag = grid2d(M.blocks[0])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    im = ax.pcolormesh(x, y, rho, cmap=SEQ, shading="nearest", rasterized=True)
    ax.contour(x, y, pmag, levels=6, colors=INK, linewidths=0.4)
    ax.axhline(0.75, color=C[1], lw=0.9, ls="--")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(r"$\rho$ (colour), magnetic pressure (contours)", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01)
    ax = axes[1]
    c = Cut.blocks[0].data
    ax.plot(c[:, 0], c[:, 1], color=C[0])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$4\pi\rho$")
    ax.set_title("cut along $y=0.75$ (dashed line)", fontsize=9)
    n = x.shape[0]
    fig.suptitle(f"Orszag-Tang vortex at $t=0.5$, ${n}^2$ cells (Sądowski et al. 2014, Fig. 6, there $640^2$)", fontsize=10)
    return fig


def fig_bondi(d):
    B1 = load(d, "bondi_1d")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    ax = axes[0]
    profs = B1.find(r"^profile")
    fine = profs[-1].data
    ax.loglog(fine[:, 0], fine[:, 2], color=INK, lw=1.0, label="Michel solution")
    b = profs[0]
    ax.loglog(b.data[:, 0], b.data[:, 1], ls="none", marker=MARK[0], ms=4, mfc="none", mew=0.9, color=C[0], label=f"koral-zig, {b.data.shape[0]} cells, $t=10$")
    ax.axvline(2.0, color=INK2, lw=0.6, ls=":")
    ax.text(2.05, ax.get_ylim()[0] * 1.3, "horizon", fontsize=7, color=INK2, rotation=90, va="bottom")
    ax.set_xlabel("$r$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("1D transonic accretion, Kerr-Schild", fontsize=9)
    ax.legend()
    ax = axes[1]
    series = []
    conv = B1.find("convergence")
    if conv:
        series.append(("1D", conv[0].data[:, 0], conv[0].data[:, 1]))
    for lab, names in (("2D through the polar axis", ("bondi_2d", "bondi_2d_fast")), ("2D magnetised (monopole)", ("bondi_mag", "bondi_mag_fast"))):
        try:
            D = load(d, *names)
            a = D.blocks[0].data
            t = re.search(r"t = (" + FLOAT + ")", D.meta.get("source", ""))
            series.append((lab + (f", $t={float(t.group(1)):g}$" if t else ""), a[:, 0], a[:, 1]))
        except Missing:
            pass
    for j, (lab, n, e) in enumerate(series):
        ax.loglog(n, e, color=C[j], marker=MARK[j], ls=DASH[j], label=f"{lab}: orders " + " ".join(f"{o:.2f}" for o in orders(n, e)))
    if series:
        n = series[0][1]
        guide(ax, n, 2.0 * series[0][2][0], 2, r"$N^{-2}$")
        ax.set_xticks(n)
        ax.set_xticklabels([f"{int(v)}" for v in n])
        ax.minorticks_off()
    ax.set_xlabel(r"radial cells $N_X$")
    ax.set_ylabel(r"$L_1$ drift of $\rho$ from the exact solution")
    ax.set_title("stationarity error", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Bondi-Michel accretion onto a Schwarzschild black hole", fontsize=10)
    return fig


def fig_alfven(d):
    D = load(d, "alfven_wave")
    a = D.blocks[0].data
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.loglog(a[:, 0], a[:, 1], color=C[0], marker=MARK[0], label="phase error after one period")
    ax.loglog(a[:, 0], np.abs(a[:, 2] - 1.0), color=C[1], marker=MARK[1], ls=DASH[1], label="amplitude error")
    guide(ax, a[:, 0], 2.0 * a[0, 1], 2, r"$N^{-2}$")
    panel_label(ax, "phase orders " + " ".join(f"{o:.2f}" for o in orders(a[:, 0], a[:, 1])), "bl")
    ax.set_xticks(a[:, 0])
    ax.set_xticklabels([f"{int(v)}" for v in a[:, 0]])
    ax.minorticks_off()
    ax.set_xlabel(r"$N_X$")
    ax.legend(loc="upper right")
    ax.set_title("Circularly polarised Alfvén wave (exact nonlinear solution)", fontsize=10)
    return fig


def fig_deflection(d):
    D = load(d, "grrt_deflection")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), constrained_layout=True)
    ax = axes[0]
    for j, b in enumerate(D.blocks):
        a = b.data
        ax.loglog(a[:, 1], a[:, 2], color=C[j], ls=DASH[j], label=b.label)
    ax.axhline(2 * np.pi, color=INK2, lw=0.6, ls=":")
    ax.text(ax.get_xlim()[1] * 0.8, 2 * np.pi * 1.15, r"$2\pi$", fontsize=7, color=INK2, ha="right")
    ax.set_xlabel(r"$b/b_c-1$")
    ax.set_ylabel(r"deflection angle $\hat\alpha$ [rad]")
    ax.set_title("exact (quadrature of the geodesic equation)", fontsize=9)
    ax.legend()
    ax = axes[1]
    labs = [r"$\epsilon=0.25$ (production)", r"$\epsilon=0.125$", r"$\epsilon=0.05$"]
    for k in range(3):
        err = np.maximum.reduce([b.data[:, 3 + k] for b in D.blocks])
        ax.loglog(D.blocks[0].data[:, 1], np.maximum(err, 1e-13), color=C[k], ls=DASH[k], label=labs[k])
    fin = np.maximum.reduce([b.data[:, 6] for b in D.blocks])
    ax.loglog(D.blocks[0].data[:, 1], fin, color=INK2, lw=0.9, ls=(0, (4, 3)), label="finite camera distance, $r=1000$")
    ax.set_xlabel(r"$b/b_c-1$")
    ax.set_ylabel("error [rad]")
    ax.set_title("integration error, worse of prograde and retrograde", fontsize=9)
    ax.legend(fontsize=7)
    src = D.meta.get("source", "")
    spin = re.search(r"a = (" + FLOAT + ")", src)
    fig.suptitle(f"Deflection of equatorial light rays in Kerr, $a={float(spin.group(1)):g}$ (Gold et al. 2020, Fig. 1)", fontsize=10)
    return fig


def fig_ipole(d):
    D = load(d, "grrt_ipole")
    b = D.blocks[0]
    n = int(D.meta["shape"].split()[0])
    ours = b.data[:, 2].reshape(n, n)
    ip = b.data[:, 3].reshape(n, n)
    fov = float(D.meta.get("fov_M", "nan"))
    half = 0.5 * fov if np.isfinite(fov) else 0.5 * n
    ext = [half, -half, -half, half]  # east to the left
    s_ours, s_ip = (float(v) for v in D.meta["flux"].split())
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), constrained_layout=True)
    vmax = max(ours.max(), ip.max())
    for ax, img, name, s in ((axes[0], ours, "koral-zig", s_ours), (axes[1], ip, "ipole", s_ip)):
        im = ax.imshow(img / vmax, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=1)
        ax.set_title(f"{name}: {s:.4f} Jy", fontsize=9)
        ax.set_xlabel(r"$x$ [$GM/c^2$]")
        ax.grid(False)
    axes[0].set_ylabel(r"$y$ [$GM/c^2$]")
    fig.colorbar(im, ax=axes[:2], pad=0.01, aspect=30, label=r"$I_\nu/I_{\nu,\max}$")
    ax = axes[2]
    diff = (ours - ip) / vmax
    lim = np.max(np.abs(diff))
    im = ax.imshow(diff, origin="lower", extent=ext, cmap=DIV, vmin=-lim, vmax=lim)
    ax.set_title(rf"difference; NMSE $={float(D.meta['nmse']):.1e}$, flux {100 * (s_ours / s_ip - 1):+.2f} %", fontsize=9)
    ax.set_xlabel(r"$x$ [$GM/c^2$]")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01, aspect=30, label=r"$(I_\mathrm{koral\text{-}zig}-I_\mathrm{ipole})/I_{\nu,\max}$")
    fig.suptitle(f"The same GRMHD snapshot imaged at 230 GHz by koral-zig and by ipole, ${n}^2$ pixels, thermal synchrotron", fontsize=10)
    return fig


FIGURES = {
    "deflection": fig_deflection,
    "ipole": fig_ipole,
    "shocktube": fig_shocktube,
    "torus": fig_torus,
    "radtubes": fig_radtubes,
    "pulse_thin": fig_pulse_thin,
    "pulse_thick": fig_pulse_thick,
    "shadow": fig_shadow,
    "atmosphere": fig_atmosphere,
    "beam": fig_beam,
    "radbondi": fig_radbondi,
    "waves": fig_waves,
    "magtubes": fig_magtubes,
    "orszag": fig_orszag,
    "bondi": fig_bondi,
    "alfven": fig_alfven,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="directory of *.dat dumps (zig build test -Dfig-dir=...)")
    ap.add_argument("--out", default=None, help="output directory (default: DIR)")
    ap.add_argument("--only", nargs="*", default=None, help="figure names to make (default: all)")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    out = args.out or args.dir
    os.makedirs(out, exist_ok=True)
    style()
    made, skipped = [], []
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        try:
            fig = fn(args.dir)
        except Missing as m:
            skipped.append(f"{name} (no {m})")
            continue
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"fig_{name}.{ext}"), dpi=args.dpi)
        plt.close(fig)
        made.append(name)
    print(f"wrote {len(made)} figures to {out}: " + ", ".join(made))
    if skipped:
        print("skipped: " + "; ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
