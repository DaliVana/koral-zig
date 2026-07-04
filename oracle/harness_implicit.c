/* M9 oracle: the implicit radiation–gas solver, exercised through the
 * full wrapper solve_implicit_lab on real PUFFY grid cells (so the rung
 * ladder, FORCEUEQPINIMPLICIT p2u, bisection start, SCALE_JACOBIAN and
 * all PUFFY RADIMP* parameters are the compiled ones).
 *
 * The successful rung (whichprim/whicheq/whichframe) and the iteration
 * count are recovered from the GLOBALINTSLOT_NIMP* / ITERIMP* counters,
 * reset before every record.
 *
 * States are the controlled sweep of harness_opac plus LTE-comoving and
 * B=0 variants; dt spans ~10 decades of kappa*dt stiffness.
 *
 * Emitted KGLD file:
 *   rad_implicit.kgld (37/18)  geom23 dt pp13 ->
 *       ret whichprim whicheq whichframe iters pp13
 *     (rung fields are -1 when ret < 0; pp then echoes the input)
 */

#include "ko.h"
#include <stdint.h>
#include <string.h>
#include <gsl/gsl_errno.h>

static FILE *open_kgld(const char *dir, const char *name,
                       uint64_t nrec, uint32_t nin, uint32_t nout)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  uint32_t version = 1;
  fwrite("KGLD", 1, 4, f);
  fwrite(&version, 4, 1, f);
  fwrite(&nrec, 8, 1, f);
  fwrite(&nin, 4, 1, f);
  fwrite(&nout, 4, 1, f);
  return f;
}

/* deterministic xorshift64*, distinct seed */
static uint64_t rngstate = 0x4b4f52414c494d50ULL; /* "KORALIMP" */
static double rnd(void)
{
  rngstate ^= rngstate >> 12;
  rngstate ^= rngstate << 25;
  rngstate ^= rngstate >> 27;
  return (double)((rngstate * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
}
static double rnd_pm(void) { return 2. * rnd() - 1.; }
static double rnd_log(double lo_exp, double hi_exp)
{
  return pow(10., lo_exp + (hi_exp - lo_exp) * rnd());
}

static void velr_gamma(ldouble gg[][5], double gam, double v[3])
{
  double n[3], qsq = 0.;
  int i, j;
  for (i = 0; i < 3; i++) n[i] = rnd_pm();
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      qsq += n[i] * n[j] * gg[i + 1][j + 1];
  double s = sqrt((gam * gam - 1.) / qsq);
  for (i = 0; i < 3; i++) v[i] = n[i] * s;
}

static int put_sym10(double *out, ldouble m[][5])
{
  int i, j, n = 0;
  for (i = 0; i < 4; i++)
    for (j = i; j < 4; j++)
      out[n++] = m[i][j];
  return n;
}

static int put_geom23(double *out, struct geometry *g)
{
  int n = put_sym10(out, g->gg);
  n += put_sym10(out + n, g->GG);
  out[n++] = g->gdet;
  out[n++] = g->alpha;
  out[n++] = g->gttpert;
  return n;
}

/* controlled gas+rad state (see harness_opac.c) */
static void controlled_state(struct geometry *geom, ldouble *pp, int variant)
{
  double v[3];
  int i;
  double rho = rnd_log(-12, -4);
  double tgas = rnd_log(5, 12);
  if (variant == 3) tgas = (rnd() < 0.5) ? rnd_log(3, 5) : rnd_log(12, 13);

  double uint_ = tgas * rho * kB_over_mugas_mp / (GAMMA - 1.);
  pp[RHO] = rho;
  pp[UU] = uint_;
  velr_gamma(geom->gg, 1. + 1.5 * rnd(), v);
  pp[VX] = v[0]; pp[VY] = v[1]; pp[VZ] = v[2];
  pp[ENTR] = calc_Sfromu(rho, uint_, geom->ix, geom->iy, geom->iz);

  if (variant == 2) {
    pp[B1] = pp[B2] = pp[B3] = 0.;
  } else {
    double beta = rnd_log(-2, 2);
    double bsq_target = 2. * (GAMMA - 1.) * uint_ / beta;
    velr_gamma(geom->gg, sqrt(1. + bsq_target), v);
    pp[B1] = v[0]; pp[B2] = v[1]; pp[B3] = v[2];
  }

  double zeta = (variant == 1) ? 1.0 : rnd_log(-2, 2);
  pp[EE] = calc_LTE_EfromT(zeta * tgas);
  if (variant == 1) {
    pp[FX] = pp[VX]; pp[FY] = pp[VY]; pp[FZ] = pp[VZ];
  } else {
    velr_gamma(geom->gg, 1. + 1.0 * rnd(), v);
    pp[FX] = v[0]; pp[FY] = v[1]; pp[FZ] = v[2];
  }

  for (i = 0; i < NV; i++)
    if (!isfinite((double)pp[i])) { fprintf(stderr, "bad state\n"); exit(1); }
}

static const int ixs[8] = {24, 48, 96, 144, 192, 240, 300, 370};
static const int iys[7] = {2, 45, 90, 179, 270, 320, 357};

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";
  int a, b, s, i;

  /* the default GSL handler aborts on an exactly singular Jacobian
   * (production KORAL would too); with it off, LU produces inf/nan and
   * the damping ladder rejects the step — the same silent path the Zig
   * invert4 takes */
  gsl_set_error_handler_off();

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  ldouble mindx, mindy, mindz, maxdt;
  set_grid(&mindx, &mindy, &mindz, &maxdt);
  calc_metric();
  alloc_loops();

  uint64_t nrec = 8 * 7 * 6;
  FILE *f = open_kgld(outdir, "rad_implicit.kgld", nrec, 37, 18);

  for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 6; s++) {
    int ix = ixs[a], iy = iys[b], iz = 0;
    struct geometry geom;
    fill_geometry(ix, iy, iz, &geom);

    ldouble pp[NV], uu[NV];
    controlled_state(&geom, pp, s % 4); /* extra generic + LTE rounds */
    ldouble dt = rnd_log(-10, 0);

    p2u(pp, uu, &geom);
    for (i = 0; i < NV; i++) {
      set_u(p, i, ix, iy, iz, pp[i]);
      set_u(u, i, ix, iy, iz, uu[i]);
    }

    for (i = 0; i < NGLOBALINTSLOT; i++) global_int_slot[i] = 0;

    int ret = solve_implicit_lab(ix, iy, iz, dt, 0);

    /* recover the successful rung from the success counters */
    int prim = -1, eq = -1, frame = -1;
    if (ret == 0) {
      if (global_int_slot[GLOBALINTSLOT_NIMPENERRAD])   { prim = RAD; eq = 0; frame = 0; }
      if (global_int_slot[GLOBALINTSLOT_NIMPENERRADFF]) { prim = RAD; eq = 0; frame = 1; }
      if (global_int_slot[GLOBALINTSLOT_NIMPENERMHD])   { prim = MHD; eq = 0; frame = 0; }
      if (global_int_slot[GLOBALINTSLOT_NIMPENERMHDFF]) { prim = MHD; eq = 0; frame = 1; }
      if (global_int_slot[GLOBALINTSLOT_NIMPENTRRAD])   { prim = RAD; eq = 1; frame = 1; }
      if (global_int_slot[GLOBALINTSLOT_NIMPENTRMHD])   { prim = MHD; eq = 1; frame = 1; }
    }
    int iters = global_int_slot[GLOBALINTSLOT_ITERIMPENERRAD] +
                global_int_slot[GLOBALINTSLOT_ITERIMPENERMHD] +
                global_int_slot[GLOBALINTSLOT_ITERIMPENTRRAD] +
                global_int_slot[GLOBALINTSLOT_ITERIMPENTRMHD];

    double in[37], out[18];
    int n = put_geom23(in, &geom);
    in[n++] = dt;
    for (i = 0; i < NV; i++) in[n++] = pp[i];

    out[0] = (double)ret;
    out[1] = (double)prim;
    out[2] = (double)eq;
    out[3] = (double)frame;
    out[4] = (double)iters;
    for (i = 0; i < NV; i++) out[5 + i] = get_u(p, i, ix, iy, iz);
    fwrite(in, 8, 37, f);
    fwrite(out, 8, 18, f);
  }
  fclose(f);

  fprintf(stderr, "harness_implicit: done\n");
  return 0;
}
