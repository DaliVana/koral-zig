/* M12 oracle: the mean-field dynamo applied once to the PUFFY t=0 state.
 *
 * The dynamo does nothing at t=0 on the real state (B³ = 0 → ΔA_φ = 0), so we
 * inject a deterministic toroidal field B³ = 3·B² into the domain (β from B³
 * ~ O(1) > BETASATURATED, exercising DAMPBETA), keep u consistent (p2u), then
 * run apply_dynamo(0, dt) — calc_avgs_throughout (scaleth) → set_bc →
 * mimic_dynamo (ΔA_φ, damping, calc_BfromA, superposition) → calc_u2p — and
 * dump the resulting B¹,B²,B³ over the domain.
 *
 * The Zig test injects the identical B³ and runs the identical apply_dynamo.
 * B¹,B² carry the ~1e-3 qags error through the field-angle / scale-height /
 * ΔA_φ inputs (M11 keystone story), so the golden matches at field-scale
 * ~1e-3, NOT a Zig error. dt = 10 makes the dynamo change ≳ that floor.
 *
 * KINI layout (see harness_init.c); domain only (gx=gy=0, stride 1).
 */

#include "ko.h"
#include <stdint.h>
#include <string.h>
#include <gsl/gsl_errno.h>

static void write_i32(FILE *f, int32_t v) { fwrite(&v, 4, 1, f); }

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

  /* ---- inject B³ = 3·B² into the domain, keep u consistent ---- */
  int ix, iy, iz, v;
  for (iz = 0; iz < NZ; iz++)
    for (iy = 0; iy < NY; iy++)
      for (ix = 0; ix < NX; ix++)
      {
        struct geometry geom;
        fill_geometry(ix, iy, iz, &geom);
        ldouble pp[NV], uu[NV];
        for (v = 0; v < NV; v++) pp[v] = get_u(p, v, ix, iy, iz);
        pp[B3] = 10.0 * pp[B2]; /* inject_factor, matches dynamo_tests.zig */
        p2u(pp, uu, &geom);
        for (v = 0; v < NV; v++) { set_u(p, v, ix, iy, iz, pp[v]); set_u(u, v, ix, iy, iz, uu[v]); }
      }

  /* ---- one dynamo application (dt = 10) ---- */
  global_dt = 10.;
  apply_dynamo(0., 10.);

  /* ---- dump B¹,B²,B³ over the domain ---- */
  char path[1024];
  snprintf(path, sizeof(path), "%s/puffy_dynamo.kini", outdir);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  uint32_t version = 1;
  fwrite("KINI", 1, 4, f);
  fwrite(&version, 4, 1, f);
  write_i32(f, NX); write_i32(f, NY); write_i32(f, NZ);
  write_i32(f, 0); write_i32(f, 0); write_i32(f, 0);
  write_i32(f, 1);
  write_i32(f, 3);
  int32_t bvars[3] = { B1, B2, B3 };
  fwrite(bvars, 4, 3, f);
  for (iz = 0; iz < NZ; iz++)
    for (iy = 0; iy < NY; iy++)
      for (ix = 0; ix < NX; ix++)
        for (v = 0; v < 3; v++)
        {
          double val = get_u(p, bvars[v], ix, iy, iz);
          fwrite(&val, 8, 1, f);
        }
  fclose(f);
  printf("wrote %s\n", path);
  return 0;
}
