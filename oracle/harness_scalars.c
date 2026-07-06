/* M13 oracle: PUFFY diagnostic scalars on the t=0 keystone state.
 *
 * Runs the same deterministic init prefix as harness_init.c up to snapshot 1
 * (prepinit + init + set_bc + calc_BfromA + set_bc — the state committed as
 * tests/golden/init/puffy_t0_p.kini.gz), then fills scaleth_otg via
 * calc_avgs_throughout() and evaluates the postproc.c reductions the Zig side
 * ports (io/scalars.zig): total mass, accretion rate through the horizon,
 * radiative + total luminosity at the outer shell, and the scale height. The
 * Zig test loads the identical primitives from puffy_t0_p and compares, so the
 * only slack is Zig's recomputed MKS2 sqrt(-g) (the two-pi spread) and libm.
 *
 * KGLD (n_in=0, n_out=5): mass, -mdot(rhorizon,0), radlum, totallum, H(15).
 */

#include "ko.h"
#include <stdint.h>
#include <gsl/gsl_errno.h>

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

  /* ---- ko.c init sequence up to snapshot 1 (matches harness_init.c) ---- */
  {
#include PR_PREPINIT
  }
  set_initial_profile();
  set_bc(0., 1);
  calc_BfromA(p, 1);
  set_bc(0., 1);

  /* fill scaleth_otg (CALCHRONTHEGO) so calc_scaleheight has data, exactly as
   * the first op_explicit / apply_dynamo would before any evolution */
  calc_avgs_throughout();

  double mass = calc_totalmass();
  double mdot = calc_mdot(rhorizonBL, 0);   /* scalars[1] = -mdot */
  double radlum = 0., totallum = 0.;
  calc_lum(5000., 1, &radlum, &totallum);
  double H = calc_scaleheight(15.);

  double out[5] = { mass, -mdot, radlum, totallum, H };

  char path[1024];
  snprintf(path, sizeof(path), "%s/puffy_scalars.kgld", outdir);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  uint32_t version = 1, nin = 0, nout = 5;
  uint64_t nrec = 1;
  fwrite("KGLD", 1, 4, f);
  fwrite(&version, 4, 1, f);
  fwrite(&nrec, 8, 1, f);
  fwrite(&nin, 4, 1, f);
  fwrite(&nout, 4, 1, f);
  fwrite(out, 8, 5, f);
  fclose(f);

  printf("harness_scalars: mass=%.10e mdot=%.10e radlum=%.10e totallum=%.10e H=%.10e\n",
         mass, -mdot, radlum, totallum, H);
  return 0;
}
