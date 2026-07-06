/* M11 oracle: the PUFFY t=0 keystone — the full deterministic init
 * sequence of ko.c:140-263 (ifinit==1):
 *
 *   PR_PREPINIT (limotorus + atmosphere + LTE split, A_phi in B slots)
 *   set_initial_profile (PR_INIT: ENTR=calc_Sfromu + p2u)
 *   set_bc                                   -> snapshot 0 (B slots = A)
 *   calc_BfromA(p,1); set_bc                 -> snapshot 1 (all primitives)
 *   PR_POSTINIT (beta normalization)         -> snapshot 2 (B slots only;
 *                                               ghosts stay unscaled, as C
 *                                               never refreshes them)
 *
 * Emitted KINI files (little-endian):
 *   char[4] "KINI", u32 version=1,
 *   i32 nx ny nz, i32 gx gy gz (ghost layers included per axis),
 *   i32 stride, i32 nvars, i32 vars[nvars],
 *   f64 data[...]: iz slowest, then iy, then ix, vars fastest;
 *   each axis runs [-g .. n+g) when its g>0, else [0 .. n) by stride.
 *
 * With EPS12_VARIANT the build's tools.c has qags epsrel 1e-8 -> 1e-12
 * (patched by gen_golden.sh) and only a stride-4 domain snapshot 1 is
 * written — it isolates the C-side quadrature tolerance in the keystone
 * diff.
 */

#include "ko.h"
#include <stdint.h>
#include <string.h>
#include <gsl/gsl_errno.h>

static void write_i32(FILE *f, int32_t v) { fwrite(&v, 4, 1, f); }

/* dump the requested primitive slots of p over the grid */
static void dump_kini(const char *dir, const char *name,
                      int gx, int gy, int stride,
                      const int32_t *vars, int nvars)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

  uint32_t version = 1;
  fwrite("KINI", 1, 4, f);
  fwrite(&version, 4, 1, f);
  write_i32(f, NX); write_i32(f, NY); write_i32(f, NZ);
  write_i32(f, gx); write_i32(f, gy); write_i32(f, 0);
  write_i32(f, stride);
  write_i32(f, nvars);
  fwrite(vars, 4, nvars, f);

  int ix, iy, iz, v;
  for (iz = 0; iz < NZ; iz++)
    for (iy = (gy ? -gy : 0); iy < NY + gy; iy += (gy ? 1 : stride))
      for (ix = (gx ? -gx : 0); ix < NX + gx; ix += (gx ? 1 : stride))
        for (v = 0; v < nvars; v++)
        {
          double val = get_u(p, vars[v], ix, iy, iz);
          fwrite(&val, 8, 1, f);
        }
  fclose(f);
  printf("wrote %s\n", path);
}

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";

  /* prad_ff2lab's u2p_rad and the qags integral run under GSL; keep the
   * same silent-error configuration as the other oracle harnesses */
  gsl_set_error_handler_off();

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  set_grid(&min_dx, &min_dy, &min_dz, &max_dt);
  alloc_loops();
  calc_metric();

  /* ---- ko.c:140 PR_PREPINIT ---- */
  {
#include PR_PREPINIT
  }

  /* ---- ko.c:211 init + ghost fill ---- */
  set_initial_profile();
  set_bc(0., 1);

  static const int32_t bvars[3] = { B1, B2, B3 };
  static int32_t allvars[NV];
  int i;
  for (i = 0; i < NV; i++) allvars[i] = i;

#ifndef EPS12_VARIANT
  /* snapshot 0: vector potential still sitting in the B slots */
  dump_kini(outdir, "puffy_t0_A.kini", NG, NG, 1, bvars, 3);
#endif

  /* ---- ko.c:242 vector potential -> B ---- */
  calc_BfromA(p, 1);
  set_bc(0., 1);

#ifdef EPS12_VARIANT
  dump_kini(outdir, "puffy_t0_eps12.kini", 0, 0, 4, allvars, NV);
  return 0;
#else
  /* snapshot 1: the full primitive state before beta normalization */
  dump_kini(outdir, "puffy_t0_p.kini", NG, NG, 1, allvars, NV);

  /* ---- ko.c:262 PR_POSTINIT (beta normalization) ---- */
  {
#include PR_POSTINIT
  }

  /* snapshot 2: scaled B (domain only is affected; ghosts stay stale) */
  dump_kini(outdir, "puffy_t0_pfinal.kini", NG, NG, 1, bvars, 3);
  return 0;
#endif
}
