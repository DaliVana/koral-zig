#!/usr/bin/env python3
"""ipolecheck.py — the GRRT snapshot cross-check against ipole.

koral-zig's imager and ipole (Moscibrodzka & Gammie 2018;
github.com/AFD-Illinois/ipole) image the SAME snapshot, cell for cell: ipole
reads KORAL dumps natively, in KORAL's own MKS2 coordinates, so nothing is
resampled. This script is the glue between the two (docs/RENDER.md,
"Snapshot cross-check against ipole"):

  pack     snap.kipl snap.h5      pack a kdmp2ipole export into the HDF5
                                  layout of ipole's KORAL reader (needs h5py)
  par      snap.h5 run.par ...    write the matching ipole parameter file
  compare  ipole.h5 ours.fits     compare the two images; with --dump DIR,
                                  write DIR/grrt_ipole.dat for paperfig.py

The whole run:

  zig build -Doptimize=ReleaseFast
  T=koral/problems/puffy/puffy_ipole.toml ; D=dumps/ipole
  ./zig-out/bin/mhdtorus   $T $D/snap.kdmp --tmax 300 --scale 1e-21
  ./zig-out/bin/kdmp2ipole $T $D/snap.kdmp $D/snap.kipl
  python3 tools/ipolecheck.py pack $D/snap.kipl $D/snap.h5
  python3 tools/ipolecheck.py par  $D/snap.h5 $D/ipole.par --size 160 --fov 60 --incl 60
  ipole -unpol -par $D/ipole.par                  # writes $D/ipole_image.h5
  ./zig-out/bin/kdmp2png $T $D/snap.kdmp $D/ours.png --nu 230 --dist 8.127 \\
      --fits $D/ours.fits --size 160 --fov 60 --incl 60 --rcam 1000 --ss 1 \\
      --sigma-cut 0 --floor-cut 0 --fcol off --eps 0.25
  python3 tools/ipolecheck.py compare $D/ipole_image.h5 $D/ours.fits --dump dumps/figdata

What is matched by construction. Both codes get the same electron density
and temperature (the ones this code derives; ipole is handed T_e in Kelvin
and a density that reproduces n_e), the same field strength (ipole's B unit
is c sqrt(4 pi rho_unit), which is KORAL's), the same black-hole mass to
the last digit (so the path length in cm agrees), and with `-unpol` the
same thermal synchrotron emissivity fit (Leung, Gammie & Noble 2011) with
absorption from Kirchhoff's law. The toml switches this code's free-free
and scattering off and the commands above switch both codes' emission cuts
off. What is NOT shared, and is therefore what the comparison tests: the
geodesic integration (ipole traces in exponential Kerr-Schild coordinates
with its own stepper), the camera, the interpolation of the snapshot along
the ray, the frame transformations to the fluid frame, the pitch angle, and
the transfer integration.
"""

import argparse
import os
import struct
import sys

import numpy as np

# ipole's constants (src/constants.h), used where its internal conversions
# must reproduce this code's numbers exactly
IP_MP = 1.67262171e-24
IP_ME = 9.1093826e-28
IP_CL = 2.99792458e10
IP_GNEWT = 6.6742e-8
IP_MSUN = 1.989e33
IP_PC = 3.085678e18

NAMES = ["r", "th", "gdet", "ne", "uint", "te", "U1", "U2", "U3", "B1", "B2", "B3", "bsq", "rho"]
SCALARS = ["t", "startx1", "startx2", "dx1", "dx2", "mksr0", "mksh0", "spin", "gam", "masscm", "rho_unit"]


def read_kipl(path):
    buf = open(path, "rb").read()
    if buf[:4] != b"KIPL":
        sys.exit(f"{path}: not a kdmp2ipole export")
    ver, n1, n2, n3 = struct.unpack("<4I", buf[4:20])
    if ver != 1:
        sys.exit(f"{path}: version {ver} not supported")
    sc = dict(zip(SCALARS, struct.unpack("<11d", buf[20:108])))
    arr = np.frombuffer(buf, dtype="<f8", offset=108).reshape(len(NAMES), n1, n2, n3)
    return sc, dict(zip(NAMES, arr)), (n1, n2, n3)


def cmd_pack(args):
    import h5py

    sc, d, (n1, n2, n3) = read_kipl(args.kipl)
    if n3 != 1:
        sys.exit("only axisymmetric (n3 = 1) snapshots are supported: ipole needs the KORAL x3 extent to be 2 pi")
    with h5py.File(args.h5, "w") as f:
        h = f.create_group("header")
        S = h5py.string_dtype("ascii", 20)
        h.create_dataset("version", data=np.bytes_("KORALv2"), dtype=h5py.string_dtype("ascii", 8))
        h.create_dataset("metric_out", data=np.bytes_("KS"), dtype=S)
        h.create_dataset("metric_run", data=np.bytes_("MKS2"), dtype=S)
        for k, v in (("n1", n1), ("n2", n2), ("n3", n3), ("has_radiation", 1), ("has_electrons", 1)):
            h.create_dataset(k, data=np.int32(v))
        h.create_dataset("bhspin", data=sc["spin"])
        h.create_dataset("gam", data=sc["gam"])
        g = h.create_group("geom")
        g.create_dataset("startx1", data=sc["startx1"])
        g.create_dataset("startx2", data=sc["startx2"])
        g.create_dataset("startx3", data=0.0)
        g.create_dataset("dx1", data=sc["dx1"])
        g.create_dataset("dx2", data=sc["dx2"])
        g.create_dataset("dx3", data=2.0 * np.pi)
        m = g.create_group("mks2")
        m.create_dataset("mksr0", data=sc["mksr0"])
        m.create_dataset("mksh0", data=sc["mksh0"])
        g.create_group("mks3")  # ipole looks into it for unsupported features
        # ipole recomputes L, T and the B unit from M_bh and the density
        # unit: M_bh is chosen so that its G M / c^2 is this code's, to the
        # last digit; the B unit c sqrt(4 pi rho_unit) is then KORAL's too.
        u = h.create_group("units")
        mbh = sc["masscm"] * IP_CL**2 / (IP_GNEWT * IP_MSUN)
        u.create_dataset("M_bh", data=mbh)
        u.create_dataset("M_unit", data=sc["rho_unit"])  # ipole reads RHO_unit from this name
        u.create_dataset("L_unit", data=sc["masscm"])
        u.create_dataset("T_unit", data=sc["masscm"] / IP_CL)
        u.create_dataset("U_unit", data=sc["rho_unit"] * IP_CL**2)
        u.create_dataset("B_unit", data=IP_CL * np.sqrt(4.0 * np.pi * sc["rho_unit"]))
        f.create_dataset("t", data=sc["t"])
        go = f.create_group("grid_out")
        go.create_dataset("r", data=d["r"])
        go.create_dataset("th", data=d["th"])
        f.create_group("grid_run").create_dataset("gdet", data=d["gdet"])
        q = f.create_group("quants")
        # ipole's n_e is rho rho_unit / (m_p + m_e): hand it the density that
        # gives THIS code's n_e (the two differ by the composition's mu_e)
        q.create_dataset("rho", data=d["ne"] * (IP_MP + IP_ME) / sc["rho_unit"])
        q.create_dataset("uint", data=d["uint"])
        q.create_dataset("te", data=d["te"])
        for k in ("U1", "U2", "U3", "B1", "B2", "B3"):
            q.create_dataset(k, data=d[k])
    print(f"wrote {args.h5}: {n1}x{n2}x{n3}, t = {sc['t']:.2f}, a = {sc['spin']}, M_bh = {mbh:.6e} Msun (ipole units)")


def cmd_par(args):
    import h5py

    with h5py.File(args.h5, "r") as f:
        masscm = float(f["header/units/L_unit"][()])
        r_out = float(np.max(f["grid_out/r"][()]))
    d_cm = args.dist * 1.0e3 * IP_PC
    fov_uas = args.fov * masscm / d_cm * 180.0 / np.pi * 3.6e9
    out = args.out or os.path.join(os.path.dirname(args.par) or ".", "ipole_image.h5")
    lines = [
        "# written by tools/ipolecheck.py par",
        f"dump {args.h5}",
        f"outfile {out}",
        f"nx {args.size}",
        f"ny {args.size}",
        f"freqcgs {args.freq * 1e9:.6e}",
        f"dsource {args.dist * 1.0e3:.6e}",
        f"rcam {args.rcam}",
        f"thetacam {args.incl}",
        "phicam 0",
        "rotcam 0",
        f"fovx_dsource {fov_uas:.10e}",
        f"fovy_dsource {fov_uas:.10e}",
        # no emission cuts: both codes image every cell of the snapshot
        "sigma_cut 1e30",
        f"rmax_geo {1.01 * r_out:.3f}",
        "counterjet 0",
        # the electron temperature comes from the dump; these only satisfy
        # ipole's reader
        "trat_small 1",
        "trat_large 1",
        f"eps {args.eps}",
        "maxnstep 100000",
        "only_unpolarized 1",
    ]
    open(args.par, "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.par}: {args.size}^2 pixels, fov {args.fov} M = {fov_uas:.3f} uas, image -> {out}")
    print(f"run:  ipole -unpol -par {args.par}")


def read_fits(path):
    """Minimal reader for render/fits.zig output (BITPIX -64, one HDU)."""
    buf = open(path, "rb").read()
    cards = {}
    for i in range(36):
        c = buf[i * 80 : (i + 1) * 80].decode("ascii")
        if c.startswith("END"):
            break
        if c[8:10] == "= ":
            cards[c[:8].strip()] = c[10:].split(" /")[0].strip()
    n1, n2 = int(cards["NAXIS1"]), int(cards["NAXIS2"])
    data = np.array(struct.unpack(f">{n1 * n2}d", buf[2880 : 2880 + 8 * n1 * n2])).reshape(n2, n1)
    return data  # [row from the bottom, column from the east (left on the sky)]; Jy / pixel


def cmd_compare(args):
    import h5py

    with h5py.File(args.ipole, "r") as f:
        ip = f["unpol"][()] * float(f["header/scale"][()])  # [i: x, j: y] -> Jy / pixel
        ftot_ip = float(f["Ftot_unpol"][()])
        # ipole stores the field of view as an angle at the camera [rad]
        cam = f["header/camera"]
        fov = float(cam["fovx"][()]) * float(cam["rcam"][()]) if "fovx" in cam else float("nan")
    ours = read_fits(args.fits)
    if ours.shape[0] != ip.shape[1] or ours.shape[1] != ip.shape[0]:
        sys.exit(f"image sizes differ: ipole {ip.shape}, ours {ours.shape}")

    # ipole stores image[i][j], i rightward and j upward on its screen; the
    # FITS rows run upward and its columns from the east. Find which of the
    # eight orientations of ipole's array is ours, and say so.
    cands = {}
    for name, a in (("[j][i]", ip.T), ("[i][j]", ip)):
        for fx in (False, True):
            for fy in (False, True):
                b = a[::-1, :] if fy else a
                b = b[:, ::-1] if fx else b
                cands[f"ipole{name}" + (" flipped in x" if fx else "") + (" flipped in y" if fy else "")] = b
    nmse = {k: float(np.sum((v - ours) ** 2) / np.sum(v**2)) for k, v in cands.items()}
    best = min(nmse, key=nmse.get)
    ipb = cands[best]
    ranked = sorted(nmse.values())

    s_ours, s_ip = float(ours.sum()), float(ipb.sum())
    peak = ipb.max()
    bright = ipb > 0.01 * peak
    rel = np.abs(ours - ipb)[bright] / ipb[bright]
    print(f"orientation: ours = {best}  (NMSE {ranked[0]:.3e}; next best {ranked[1]:.3e})")
    print(f"total flux:  koral-zig {s_ours:.6f} Jy   ipole {s_ip:.6f} Jy (its own sum {ftot_ip:.6f})   difference {100 * (s_ours / s_ip - 1):+.3f} %")
    print(f"NMSE (I):    {ranked[0]:.3e}")
    print(f"pixels above 1 % of the peak: median |dI|/I {100 * np.median(rel):.3f} %, 95th percentile {100 * np.percentile(rel, 95):.3f} %")
    print(f"peak:        koral-zig {ours.max():.4e}  ipole {peak:.4e} Jy/pixel ({100 * (ours.max() / peak - 1):+.3f} %)")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        n = ours.shape[0]
        path = os.path.join(args.dump, "grrt_ipole.dat")
        with open(path, "w") as f:
            f.write("# source = GRRT snapshot cross-check: koral-zig kdmp2png against ipole -unpol, same snapshot\n")
            f.write("# columns = column(east->west) row(south->north) I_koralzig I_ipole   [Jy / pixel]\n")
            f.write(f"# shape = {n} {n}\n")
            f.write(f"# fov_M = {fov}\n")
            f.write(f"# flux = {s_ours:.8e} {s_ip:.8e}\n")
            f.write(f"# nmse = {ranked[0]:.6e}\n")
            f.write(f"# orientation = {best}\n")
            for r in range(n):
                for c in range(n):
                    f.write(f"{c} {r} {ours[r, c]:.10e} {ipb[r, c]:.10e}\n")
        print(f"wrote {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="kdmp2ipole export -> KORAL-format HDF5 for ipole")
    p.add_argument("kipl")
    p.add_argument("h5")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("par", help="write the ipole parameter file matching a kdmp2png command")
    p.add_argument("h5")
    p.add_argument("par")
    p.add_argument("--out", default=None, help="ipole image file (default: next to the par file)")
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--fov", type=float, default=60.0, help="field of view in GM/c^2")
    p.add_argument("--incl", type=float, default=60.0)
    p.add_argument("--freq", type=float, default=230.0, help="GHz")
    p.add_argument("--dist", type=float, default=8.127, help="kpc")
    p.add_argument("--rcam", type=float, default=1000.0)
    p.add_argument("--eps", type=float, default=0.01, help="ipole's geodesic step parameter")
    p.set_defaults(fn=cmd_par)
    p = sub.add_parser("compare", help="compare an ipole image with a kdmp2png FITS image")
    p.add_argument("ipole")
    p.add_argument("fits")
    p.add_argument("--dump", default=None, help="directory for grrt_ipole.dat (tools/paperfig.py)")
    p.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
