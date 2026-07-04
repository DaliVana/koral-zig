#!/bin/bash
# Regenerate C-oracle golden files from ../koral_lite.
#
# Copies the koral_lite sources into oracle/build/<variant>/src (never
# touches the original tree), patches PROBLEM, compiles the compute objects
# (no silo, no MPI, no HDF5 -- all ifdef'd out) with clang + GSL, builds the
# harnesses, and runs them into tests/golden/.
#
# Variants:
#   puffy    (PROBLEM 147) -> harness_metric/state/flux/rad/opac -> metric/ state/ flux/ rad/
#   zigsod   (PROBLEM 200) -> harness_step               -> step/sod64.kstp
#   zigot    (PROBLEM 201) -> harness_step               -> step/ot32.kstp + flux/ct.kgld + flux/bfroma.kgld
#   zigmhdtube (PROBLEM 202) -> harness_step             -> step/mhdtube64.kstp
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

echo "== [puffy] running harnesses"
mkdir -p "$ROOT/tests/golden/metric" "$ROOT/tests/golden/state" "$ROOT/tests/golden/flux" "$ROOT/tests/golden/rad"
cd "$BUILD/puffy/src"
./harness_metric "$ROOT/tests/golden/metric"
./harness_state "$ROOT/tests/golden/state"
./harness_flux "$ROOT/tests/golden/flux"
./harness_rad "$ROOT/tests/golden/rad"
./harness_opac "$ROOT/tests/golden/rad"

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

echo "== writing manifest"
SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$ROOT/tests/golden/manifest.json" <<EOF
{
  "koral_lite_sha": "$SHA",
  "problem": [147, 200, 201, 202],
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
    "flux/ct.kgld": "facedim ix iy fB1 fB2 fB3 -> fB1' fB2' fB3' (flux_ct, ZIGOT 32x32)",
    "flux/bfroma.kgld": "ix iy A1 A2 A3 -> B1 B2 B3 (calc_BfromA overwrite, ZIGOT periodic)",
    "step/sod64.kstp": "ZIGSOD (200): 64x1 SR Sod, MINK PPM RK2IMEX LAXF, 10 steps",
    "step/ot32.kstp": "ZIGOT (201): 32x32 SR Orszag-Tang, periodic, VECPOTGIVEN, 10 steps",
    "step/mhdtube64.kstp": "ZIGMHDTUBE (202): 64x1 Balsara-1 (Gamma=2), 10 steps"
  }
}
EOF

echo "== done"
