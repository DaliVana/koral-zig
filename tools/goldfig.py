#!/usr/bin/env python3
"""goldfig.py — reproduce Gold et al. 2020 (ApJ 897, 148) Figures 2 & 3
for koral-zig.

Row 1 (their Figure 2): the five standardized test images, shown as
S/S_tot,exact on a linear 0..2e-4 scale, cubehelix colormap, relative
RA/DEC in uas (West right), with the koral-zig total flux annotated.

Row 2 (their Figure 3, honest variant): per-pixel difference dS [Jy/pixel]
after convolving with a 20 uas FWHM beam. The paper differences each code
against its arbitrary-precision EXACT image, which is not published as
data; the equivalent internal diagnostic is production parameters minus a
step-converged reference (eps 0.25 vs 0.05) — the integration-error map.
Shared camera/pixelization systematics (the paper's dominant cross-code
discrepancy source, ~+1% of total flux for the ipole/RAPTOR class and for
koral-zig alike) cancel in this difference; the total-flux comparison
against EXACT lives in `goldtest`'s table.

usage:  goldfig.py DIR [--out FILE.png]
        (DIR holds gold_testN.fits and gold_testN_ref.fits from
         `goldtest DIR --fits [--tag ref]`)
"""

import argparse
import struct
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# their Table 2 EXACT fluxes [Jy] and Table 1 scene labels
S_EXACT = [1.6465, 1.4360, 0.4418, 0.2710, 0.0255]
UAS_PER_M = 6.0e11 / 2.4e22 * 180.0 / np.pi * 3.6e9  # ~5.157
HALF_FOV_UAS = 15.0 * UAS_PER_M
BEAM_FWHM_UAS = 20.0


def read_fits(path):
    """Minimal reader for render/fits.zig output (BITPIX -64, one HDU)."""
    with open(path, "rb") as f:
        buf = f.read()
    cards = {}
    for i in range(36):
        c = buf[i * 80 : (i + 1) * 80].decode("ascii")
        if c.startswith("END"):
            break
        if c[8:10] == "= ":
            cards[c[:8].strip()] = c[10:].split(" /")[0].strip()
    n1, n2 = int(cards["NAXIS1"]), int(cards["NAXIS2"])
    data = np.array(
        struct.unpack(f">{n1 * n2}d", buf[2880 : 2880 + 8 * n1 * n2])
    ).reshape(n2, n1)
    return data  # row 0 = bottom (-DEC), col 0 = +RA(east); Jy/pixel


def beam_convolve(img, fwhm_px):
    """Gaussian beam via FFT (periodic edges are fine: the frame edge is dim)."""
    n = img.shape[0]
    sigma = fwhm_px / 2.3548
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k)
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (kx**2 + ky**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * kernel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{args.dir}/goldfig.png"

    prod = [read_fits(f"{args.dir}/gold_test{i + 1}.fits") for i in range(5)]
    ref = [read_fits(f"{args.dir}/gold_test{i + 1}_ref.fits") for i in range(5)]
    n = prod[0].shape[0]
    pix_uas = 2.0 * HALF_FOV_UAS / n

    plt.rcParams.update(
        {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9}
    )
    fig, axes = plt.subplots(
        2, 5, figsize=(13.2, 6.0), sharex=True, sharey=True, constrained_layout=True
    )
    extent = [HALF_FOV_UAS, -HALF_FOV_UAS, -HALF_FOV_UAS, HALF_FOV_UAS]

    # ---- row 1: Figure 2 (S / S_tot,exact) ----
    for i, ax in enumerate(axes[0]):
        im1 = ax.imshow(
            prod[i] / S_EXACT[i],
            origin="lower",
            extent=extent,
            cmap="cubehelix",
            vmin=0.0,
            vmax=2.0e-4,
        )
        ax.set_title(f"Test {i + 1}", fontsize=11)
        ax.text(
            0.05,
            0.92,
            rf"$S_\mathrm{{tot}} = {np.sum(prod[i]):.4f}\,$Jy",
            transform=ax.transAxes,
            color="k",
            fontsize=8,
        )
        ax.text(
            0.05,
            0.06,
            "KORAL-ZIG",
            transform=ax.transAxes,
            color="w",
            fontsize=9,
            style="italic",
        )
    cb1 = fig.colorbar(
        im1, ax=axes[0], pad=0.01, aspect=30, ticks=np.arange(0, 2.1e-4, 2.5e-5)
    )
    cb1.set_label(r"$S/S_\mathrm{tot,exact}$")
    cb1.formatter.set_powerlimits((0, 0))
    cb1.update_ticks()

    # ---- row 2: Figure 3 analog (beam-convolved dS, production - reference) ----
    fwhm_px = BEAM_FWHM_UAS / pix_uas
    norm = SymLogNorm(linthresh=1e-8, vmin=-2e-6, vmax=2e-6, base=10)
    for i, ax in enumerate(axes[1]):
        d = beam_convolve(prod[i] - ref[i], fwhm_px)
        im2 = ax.imshow(d, origin="lower", extent=extent, cmap="RdBu_r", norm=norm)
        mse = np.sum((prod[i] - ref[i]) ** 2) / np.sum(prod[i] ** 2)
        ax.text(
            0.05,
            0.92,
            rf"MSE $= {mse:.1e}$",
            transform=ax.transAxes,
            color="k",
            fontsize=8,
        )
        ax.text(
            0.05,
            0.06,
            r"$\epsilon\,0.25 - \epsilon\,0.05$",
            transform=ax.transAxes,
            color="k",
            fontsize=8,
        )
    cb2 = fig.colorbar(im2, ax=axes[1], pad=0.01, aspect=30)
    cb2.set_label(r"$\Delta S$ [Jy/pixel]")

    for ax in axes[1]:
        ax.set_xlabel(r"Relative R.A. [$\mu$as]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Relative DEC [$\mu$as]")

    fig.suptitle(
        "koral-zig on the Gold et al. 2020 standardized EHT GRRT tests\n"
        "top: images, their Fig. 2 normalization  |  bottom: step-convergence "
        f"error map, their Fig. 3 style ({BEAM_FWHM_UAS:.0f} $\\mu$as beam)",
        fontsize=10,
    )
    fig.savefig(out, dpi=170)
    print(f"wrote {out}  ({n}x{n} px, {pix_uas:.3f} uas/px)")


if __name__ == "__main__":
    sys.exit(main())
