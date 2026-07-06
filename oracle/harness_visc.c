/* M12 oracle: the radiative shear-viscosity tensor on the PUFFY t=0 state.
 *
 * Runs the identical deterministic init as harness_init (through PR_POSTINIT),
 * then — exactly as problem.c:127 does at the start of every step — calls
 * calc_Rij_visc_total() with a fixed global_dt, and dumps two goldens over the
 * domain plus a one-cell ghost ring:
 *
 *   puffy_visc.kini  : Rijviscglobal, the per-cell viscous R^i_j (16 comps).
 *                      = indices_2221(-2 nu Ehat_rest sigma^ij). nu (via
 *                      calc_chi) and Ehat carry C's ~1e-3 qags error (like the
 *                      M11 keystone), so this matches at field-scale ~1e-3.
 *   puffy_shear.kini : sigma^ij (16 comps) + nu (1 comp) = 17 per cell, from
 *                      calc_rad_shearviscosity. sigma^ij depends only on the
 *                      (analytic, 1e-13) radiation velocities → isolates the
 *                      new FD/Christoffel code; nu isolates calc_chi + the mfp
 *                      and nudamp limiters.
 *
 * global_dt = 1.0 is fixed so the RADVISCNUDAMP cap (nulimit = mindx^2/(4 dt))
 * is deterministic; the Zig side passes the same value. Both goldens are
 * before the face average and the RADVISCMAXVELDAMP velocity cap.
 *
 * KINI layout (see harness_init.c); gx = gy = 1. Corner ghosts are pre-zeroed
 * and skipped by calc_Rij_visc_total (and by the shear loop), so they read 0
 * on both sides.
 */

#include "ko.h"
#include <stdint.h>
#include <string.h>
#include <gsl/gsl_errno.h>

static void write_i32(FILE *f, int32_t v) { fwrite(&v, 4, 1, f); }

/* skip corner ghosts (C: if_outsidegc == 1) */
static int outside_gc(int ix, int iy, int iz)
{
  int n = 0;
  if (ix < 0 || ix >= NX) n++;
  if (iy < 0 || iy >= NY) n++;
  if (iz < 0 || iz >= NZ) n++;
  return n >= 2;
}

static FILE *open_kini(const char *dir, const char *name, int g, int nvars)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  uint32_t version = 1;
  fwrite("KINI", 1, 4, f);
  fwrite(&version, 4, 1, f);
  write_i32(f, NX); write_i32(f, NY); write_i32(f, NZ);
  write_i32(f, g); write_i32(f, g); write_i32(f, 0);
  write_i32(f, 1);
  write_i32(f, nvars);
  for (int k = 0; k < nvars; k++) write_i32(f, k);
  return f;
}

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";

  gsl_set_error_handler_off();

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  set_grid(&min_dx, &min_dy, &min_dz, &max_dt);
  alloc_loops();
  calc_metric();

  /* ---- full deterministic init (ko.c:140-263) ---- */
  {
#include PR_PREPINIT
  }
  set_initial_profile();
  set_bc(0., 1);
  calc_BfromA(p, 1);
  set_bc(0., 1);
  {
#include PR_POSTINIT
  }
  /* NO set_bc after postinit: the real first-step calc_Rij_visc_total
   * (problem.c:127) runs before any set_bc, so ghost B stays unscaled — the
   * "stale ghosts" state the M11 pfinal keystone documented. Zig's initAll
   * leaves exactly the same state. */

  /* ---- calc_Rij_visc_total with a fixed global_dt (problem.c:114+127) ---- */
  global_dt = 1.0;

  int ix, iy, iz, i, j, v;
  /* pre-zero the dump range so corner ghosts (never written) read 0 */
  for (iy = -1; iy < NY + 1; iy++)
    for (ix = -1; ix < NX + 1; ix++)
      for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
          set_Tfull(Rijviscglobal, i, j, ix, iy, 0, 0.);

  calc_Rij_visc_total();

  /* ---- dump Rijviscglobal (R^i_j) ---- */
  FILE *fv = open_kini(outdir, "puffy_visc.kini", 1, 16);
  for (iz = 0; iz < NZ; iz++)
    for (iy = -1; iy < NY + 1; iy++)
      for (ix = -1; ix < NX + 1; ix++)
        for (i = 0; i < 4; i++)
          for (j = 0; j < 4; j++)
          {
            double val = get_Tfull(Rijviscglobal, i, j, ix, iy, iz);
            fwrite(&val, 8, 1, fv);
          }
  fclose(fv);
  printf("wrote %s/puffy_visc.kini\n", outdir);

  /* ---- dump sigma^ij + nu from calc_rad_shearviscosity ---- */
  FILE *fs = open_kini(outdir, "puffy_shear.kini", 1, 17);
  for (iz = 0; iz < NZ; iz++)
    for (iy = -1; iy < NY + 1; iy++)
      for (ix = -1; ix < NX + 1; ix++)
      {
        double buf[17];
        for (v = 0; v < 17; v++) buf[v] = 0.;
        if (!outside_gc(ix, iy, iz))
        {
          struct geometry geom;
          fill_geometry(ix, iy, iz, &geom);
          ldouble pp[NV];
          for (v = 0; v < NV; v++) pp[v] = get_u(p, v, ix, iy, iz);
          ldouble shear[4][4], nu;
          int derdir[3] = {0, 0, 0};
          calc_rad_shearviscosity(pp, &geom, shear, &nu, derdir);
          for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
              buf[i * 4 + j] = shear[i][j];
          buf[16] = nu;
        }
        fwrite(buf, 8, 17, fs);
      }
  fclose(fs);
  printf("wrote %s/puffy_shear.kini\n", outdir);

  return 0;
}
