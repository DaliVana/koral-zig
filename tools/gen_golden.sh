#!/bin/bash
# Regenerate C-oracle golden files from ../koral_lite.
#
# Copies the koral_lite sources into oracle/build/<variant>/src (never
# touches the original tree), patches PROBLEM, compiles the compute objects
# (no silo, no MPI, no HDF5 -- all ifdef'd out) with clang + GSL, builds the
# harnesses, and runs them into tests/golden/.
#
# Variants:
#   puffy    (PROBLEM 147) -> harness_metric/state/flux/rad/opac/implicit/init/scalars -> metric/ state/ flux/ rad/ init/
#   puffyeps12 (PROBLEM 147, qags epsrel 1e-12) -> harness_init -> init/puffy_t0_eps12
#   zigpuffy (PROBLEM 147, TNX/TNY 64/60)  -> harness_puffy_step -> step/puffy64.kstp.gz
#   zigsod   (PROBLEM 200) -> harness_step               -> step/sod64.kstp
#   zigot    (PROBLEM 201) -> harness_step               -> step/ot32.kstp + flux/ct.kgld + flux/bfroma.kgld
#   zigmhdtube (PROBLEM 202) -> harness_step             -> step/mhdtube64.kstp
#   zigradtube (PROBLEM 203) -> harness_step             -> step/radtube64.kstp
#   zigradpulse (PROBLEM 204) -> harness_step            -> step/radpulse64.kstp
#
# Usage: tools/gen_golden.sh            (KORAL_LITE=<path> to override)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${KORAL_LITE:-$ROOT/../koral_lite}"
BUILD="$ROOT/oracle/build"
GSL_PREFIX="$(brew --prefix gsl)"

[ -f "$SRC/ko.h" ] || { echo "koral_lite not found at $SRC" >&2; exit 1; }

OBJS="mpi u2prad magn postproc fileop misc physics finite problem metric relele rad opacities u2p u2p_ff frames p2u nonthermal"
CFLAGS="-O2 -fcommon -w -I$GSL_PREFIX/include"

# prepare_variant <dir> <problem-number>
prepare_variant() {
  local dir="$1" prob="$2"
  echo "== [$dir] copying sources (PROBLEM $prob)"
  rm -rf "$BUILD/$dir"
  mkdir -p "$BUILD/$dir/src"
  cp "$SRC"/*.c "$SRC"/*.h "$BUILD/$dir/src/"
  cp -R "$SRC/PROBLEMS" "$BUILD/$dir/src/PROBLEMS"
  cp -R "$ROOT/oracle/problems/"* "$BUILD/$dir/src/PROBLEMS/"

  # register the koral-zig test problems in problem.h
  python3 - "$BUILD/$dir/src/problem.h" <<'EOF'
import sys
path = sys.argv[1]
text = open(path).read()
block = """
#if(PROBLEM==200)
#define PR_DEFINE "PROBLEMS/ZIGSOD/define.h"
#define PR_BC "PROBLEMS/ZIGSOD/bc.c"
#define PR_INIT "PROBLEMS/ZIGSOD/init.c"
#endif

#if(PROBLEM==201)
#define PR_DEFINE "PROBLEMS/ZIGOT/define.h"
#define PR_BC "PROBLEMS/ZIGOT/bc.c"
#define PR_INIT "PROBLEMS/ZIGOT/init.c"
#endif

#if(PROBLEM==202)
#define PR_DEFINE "PROBLEMS/ZIGMHDTUBE/define.h"
#define PR_BC "PROBLEMS/ZIGMHDTUBE/bc.c"
#define PR_INIT "PROBLEMS/ZIGMHDTUBE/init.c"
#endif

#if(PROBLEM==203)
#define PR_DEFINE "PROBLEMS/ZIGRADTUBE/define.h"
#define PR_BC "PROBLEMS/ZIGRADTUBE/bc.c"
#define PR_INIT "PROBLEMS/ZIGRADTUBE/init.c"
#define PR_KAPPA "PROBLEMS/ZIGRADTUBE/kappa.c"
#endif

#if(PROBLEM==204)
#define PR_DEFINE "PROBLEMS/ZIGRADPULSE/define.h"
#define PR_BC "PROBLEMS/ZIGRADPULSE/bc.c"
#define PR_INIT "PROBLEMS/ZIGRADPULSE/init.c"
#define PR_KAPPA "PROBLEMS/ZIGRADPULSE/kappa.c"
#endif

"""
anchor = "/*********************/\n//including problem specific definitions from PROBLEMS/XXX/define.h"
assert anchor in text, "problem.h anchor not found"
open(path, "w").write(text.replace(anchor, block + anchor))
EOF

  sed -i '' "s/^#define PROBLEM .*/#define PROBLEM $prob/" "$BUILD/$dir/src/problem.h"
  grep -q "^#define PROBLEM $prob" "$BUILD/$dir/src/problem.h"
}

compile_objs() {
  local dir="$1"
  echo "== [$dir] compiling (clang, serial, GSL only)"
  cd "$BUILD/$dir/src"
  for o in $OBJS; do
    cc $CFLAGS -c "$o.c" -o "$o.o" &
  done
  wait
}

build_harness() {
  local dir="$1" name="$2"
  cp "$ROOT/oracle/$name.c" "$BUILD/$dir/src/"
  cd "$BUILD/$dir/src"
  cc $CFLAGS -c "$name.c" -o "$name.o"
  cc -o "$name" "$name.o" $(for o in $OBJS; do echo "$o.o"; done) \
     -L"$GSL_PREFIX/lib" -lgsl -lgslcblas -lm
}

# ---- PUFFY (147): metric/state/flux function goldens -----------------------
prepare_variant puffy 147
compile_objs puffy
build_harness puffy harness_metric
build_harness puffy harness_state
build_harness puffy harness_flux
build_harness puffy harness_rad
build_harness puffy harness_opac
build_harness puffy harness_implicit
build_harness puffy harness_init
build_harness puffy harness_visc
build_harness puffy harness_dynamo
build_harness puffy harness_scalars
build_harness puffy harness_bench_implicit # timing only — run by hand, no golden

echo "== [puffy] running harnesses"
mkdir -p "$ROOT/tests/golden/metric" "$ROOT/tests/golden/state" "$ROOT/tests/golden/flux" "$ROOT/tests/golden/rad" "$ROOT/tests/golden/init"
cd "$BUILD/puffy/src"
./harness_metric "$ROOT/tests/golden/metric"
./harness_state "$ROOT/tests/golden/state"
./harness_flux "$ROOT/tests/golden/flux"
./harness_rad "$ROOT/tests/golden/rad"
./harness_opac "$ROOT/tests/golden/rad"
./harness_implicit "$ROOT/tests/golden/rad"
./harness_scalars "$ROOT/tests/golden/init"

# ---- PUFFY t=0 keystone (harness_init on the puffy build) -------------------
echo "== [puffy] running harness_init (keystone t=0)"
./harness_init "$ROOT/tests/golden/init"
# the full-grid snapshots are large (~15 MB); gzip for the repo
for kf in puffy_t0_A puffy_t0_p puffy_t0_pfinal; do
  gzip -9 -f "$ROOT/tests/golden/init/$kf.kini"
done

# ---- PUFFY radiative shear-viscosity tensor (M12) --------------------------
echo "== [puffy] running harness_visc (radviscosity R^i_j + shear at t=0)"
./harness_visc "$ROOT/tests/golden/init"
gzip -9 -f "$ROOT/tests/golden/init/puffy_visc.kini"
gzip -9 -f "$ROOT/tests/golden/init/puffy_shear.kini"

# ---- PUFFY dynamo (M12): apply_dynamo on injected toroidal field ----------
echo "== [puffy] running harness_dynamo (mimic_dynamo on B³=10·B²)"
./harness_dynamo "$ROOT/tests/golden/init"
gzip -9 -f "$ROOT/tests/golden/init/puffy_dynamo.kini"

# ---- PUFFY t=0 epsrel-1e-12 variant (isolates C's qags tolerance) ----------
# same problem, but tools.c qags epsrel tightened 1e-8 -> 1e-12; only the
# stride-4 domain snapshot is written (attribution, not a hard gate)
prepare_variant puffyeps12 147
sed -i '' 's/LT_RIN, rml, 0., 1\.e-8,/LT_RIN, rml, 0., 1.e-12,/' \
  "$BUILD/puffyeps12/src/PROBLEMS/PUFFY/tools.c"
grep -q "1.e-12, worksize" "$BUILD/puffyeps12/src/PROBLEMS/PUFFY/tools.c"
compile_objs puffyeps12
# compile harness_init with EPS12_VARIANT for this build
cp "$ROOT/oracle/harness_init.c" "$BUILD/puffyeps12/src/"
cd "$BUILD/puffyeps12/src"
cc $CFLAGS -DEPS12_VARIANT -c harness_init.c -o harness_init.o
cc -o harness_init harness_init.o $(for o in $OBJS; do echo "$o.o"; done) \
   -L"$GSL_PREFIX/lib" -lgsl -lgslcblas -lm
./harness_init "$ROOT/tests/golden/init"
gzip -9 -f "$ROOT/tests/golden/init/puffy_t0_eps12.kini"

# ---- step-test variants -----------------------------------------------------
mkdir -p "$ROOT/tests/golden/step"

prepare_variant zigsod 200
compile_objs zigsod
build_harness zigsod harness_step
cd "$BUILD/zigsod/src"
./harness_step "$ROOT/tests/golden/step" sod64.kstp 10

prepare_variant zigot 201
compile_objs zigot
build_harness zigot harness_step
cd "$BUILD/zigot/src"
./harness_step "$ROOT/tests/golden/step" ot32.kstp 10
mv "$ROOT/tests/golden/step/ct.kgld" "$ROOT/tests/golden/flux/ct.kgld" 2>/dev/null || true
mv "$ROOT/tests/golden/step/bfroma.kgld" "$ROOT/tests/golden/flux/bfroma.kgld" 2>/dev/null || true

prepare_variant zigmhdtube 202
compile_objs zigmhdtube
build_harness zigmhdtube harness_step
cd "$BUILD/zigmhdtube/src"
./harness_step "$ROOT/tests/golden/step" mhdtube64.kstp 10

prepare_variant zigradtube 203
compile_objs zigradtube
build_harness zigradtube harness_step
cd "$BUILD/zigradtube/src"
./harness_step "$ROOT/tests/golden/step" radtube64.kstp 10

prepare_variant zigradpulse 204
compile_objs zigradpulse
build_harness zigradpulse harness_step
cd "$BUILD/zigradpulse/src"
./harness_step "$ROOT/tests/golden/step" radpulse64.kstp 10

# ---- PUFFY full-pipeline step test (M13) -----------------------------------
# reduced-grid PUFFY (TNX 384->64, TNY 360->60) so the KSTP golden is
# committable; the coordinate extents + all physics stay PUFFY's. Full init
# (limotorus + beta-norm) then 4 forced-dt RK2IMEX steps.
prepare_variant zigpuffy 147
sed -i '' 's/^#define TNX .*/#define TNX 64/' "$BUILD/zigpuffy/src/PROBLEMS/PUFFY/define.h"
sed -i '' 's/^#define TNY .*/#define TNY 60/' "$BUILD/zigpuffy/src/PROBLEMS/PUFFY/define.h"
grep -q "^#define TNX 64" "$BUILD/zigpuffy/src/PROBLEMS/PUFFY/define.h"
grep -q "^#define TNY 60" "$BUILD/zigpuffy/src/PROBLEMS/PUFFY/define.h"
compile_objs zigpuffy
build_harness zigpuffy harness_puffy_step
cd "$BUILD/zigpuffy/src"
./harness_puffy_step "$ROOT/tests/golden/step" puffy64.kstp 4
gzip -9 -f "$ROOT/tests/golden/step/puffy64.kstp"

echo "== writing manifest"
SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$ROOT/tests/golden/manifest.json" <<EOF
{
  "koral_lite_sha": "$SHA",
  "problem": [147, 200, 201, 202, 203, 204],
  "compiler": "$(cc --version | head -1)",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "files": {
    "metric/metric_points.kgld": "coords,x0..x3 -> gg16 GG16 gdet dlgdet3 Kr64 gttpert",
    "metric/coco_dxdx.kgld": "co1,co2,x0..x3 -> xout4 dxdx16",
    "metric/krzysie_grid.kgld": "ix,iy -> x1 x2 Kr64 (MODYFIKUJKRZYSIE-corrected, PUFFY grid)",
    "state/relele_convvels.kgld": "gg10 GG10 w1 w2 v3 -> u2[4] (all 9 vel pairs)",
    "state/relele_boost.kgld": "gg10 GG10 vel3 A4 -> L16 boost2_lab2ff4 boost2_ff2lab4",
    "state/frames_transpall.kgld": "co1 co2 x3 pp13 geom1_20 geom2_20 -> pp2[13]",
    "state/physics_tij.kgld": "gg10 GG10 pp13 -> Tij16 ucon4 ucov4 bcon4 bcov4 bsq",
    "state/p2u.kgld": "geom23 pp13 -> uu13",
    "state/u2p_solver.kgld": "geom23 uu13 guess13 -> ret_hot pp_hot9 ret_entr pp_entr9",
    "state/floors.kgld": "geom23 pp13 -> ret pp13 (check_floors_mhd, DRIFTFRAME)",
    "flux/recon.kgld": "u5 dx5 param theta -> ul ur (avg2point, INT_ORDER=2 base)",
    "flux/wavespeed.kgld": "geom23 pp13 -> ahd xl,xr,yl,yr (gas wavespeeds)",
    "flux/fluxprime.kgld": "geom23(face) idim pp13 -> ff13 (f_flux_prime, Rijvisc=0)",
    "rad/rad_rij.kgld": "gg10 GG10 pp13 -> Rij16 Rtt ucon4 (calc_Rij_M1 + calc_ff_Rtt)",
    "rad/rad_u2prad.kgld": "geom23 uu4 guess4 -> ret cor pp4 (u2p_rad incl. cold branches)",
    "rad/rad_wavespeeds.kgld": "gg10 GG10 pp13 tau3 -> axl0 axr0 ayl0 ayr0 axl axr ayl ayr",
    "rad/rad_floors.kgld": "geom23 pp13 -> ret pp13 (check_floors_rad, PUFFY ratios)",
    "rad/rad_thermo.kgld": "geom23 pp13 -> Tgas Te ne Ehat TradBB kappaes opac6 kappa (fill_struct_of_state)",
    "rad/rad_opac.kgld": "geom23 pp13 -> kappa kappaes chi (standalone entry points, Trad=Te kappaes)",
    "rad/rad_gi.kgld": "geom23 pp13 -> Gi_lab4 Gi_ff4 (calc_Gi thermal + Comptonization)",
    "rad/rad_implicit.kgld": "geom23 dt pp13 -> ret prim eq frame iters pp13 (solve_implicit_lab, full ladder)",
    "flux/ct.kgld": "facedim ix iy fB1 fB2 fB3 -> fB1' fB2' fB3' (flux_ct, ZIGOT 32x32)",
    "flux/bfroma.kgld": "ix iy A1 A2 A3 -> B1 B2 B3 (calc_BfromA overwrite, ZIGOT periodic)",
    "step/sod64.kstp": "ZIGSOD (200): 64x1 SR Sod, MINK PPM RK2IMEX LAXF, 10 steps",
    "step/ot32.kstp": "ZIGOT (201): 32x32 SR Orszag-Tang, periodic, VECPOTGIVEN, 10 steps",
    "step/mhdtube64.kstp": "ZIGMHDTUBE (202): 64x1 Balsara-1 (Gamma=2), 10 steps",
    "step/radtube64.kstp": "ZIGRADTUBE (203): 64x1 Farris tube 2, grey kappa=0.2rho, PUFFY implicit block, Dirichlet BC, 10 steps (4 flags)",
    "step/radpulse64.kstp": "ZIGRADPULSE (204): 64x1 optically thick LTE pulse, grey kappa=10rho, 10 steps (4 flags)",
    "init/puffy_t0_A.kini.gz": "PUFFY (147) t=0 keystone snapshot 0: A_phi in B slots after set_bc, full 384x360 + ghosts",
    "init/puffy_t0_p.kini.gz": "PUFFY (147) t=0 keystone snapshot 1: all 13 primitives after calc_BfromA + set_bc, full grid + ghosts",
    "init/puffy_t0_pfinal.kini.gz": "PUFFY (147) t=0 keystone snapshot 2: beta-normalized B slots (domain scaled, ghosts stale)",
    "init/puffy_t0_eps12.kini.gz": "PUFFY (147) t=0 with qags epsrel 1e-12: stride-4 domain, all 13 prims (quadrature attribution)",
    "init/puffy_visc.kini.gz": "PUFFY (147) t=0 viscous R^i_j (calc_Rij_visc_total, global_dt=1): 16 tensor comps, domain + 1 ghost ring",
    "init/puffy_shear.kini.gz": "PUFFY (147) t=0 shear sigma^ij + nu (calc_rad_shearviscosity): 17 comps, domain + 1 ghost ring (isolates FD/Christoffel + calc_chi)",
    "init/puffy_dynamo.kini.gz": "PUFFY (147) apply_dynamo on injected B³=10·B² (dt=10): B1,B2,B3 over the domain (scaleth + field-angle + ΔA_φ + calc_BfromA + DAMPBETA)",
    "init/puffy_scalars.kgld": "PUFFY (147) t=0 diagnostic scalars (calc_scalars on the snapshot-1 state): mass, -mdot(rhorizon,0), radlum, totallum, H(15)",
    "step/puffy64.kstp.gz": "PUFFY (147) reduced 64x60 full pipeline (limotorus init + beta-norm), 4 forced-dt RK2IMEX steps: implicit rad source + radviscosity + dynamo + polar-axis (4 flags)"
  }
}
EOF

echo "== done"
