/* Timing companion to harness_implicit.c: generates the SAME deterministic
 * battery (same xorshift64* seed, same cell/variant sweep — so the same
 * 336 records that rad_implicit.kgld holds and tools/bench_implicit.zig
 * replays), then times R rounds of solve_implicit_lab over it.
 *
 * Usage: harness_bench_implicit [rounds]     (default 30; round 0 = warmup)
 */

#include "ko.h"
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_errno.h>

/* deterministic xorshift64*, distinct seed (identical to harness_implicit) */
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

/* controlled gas+rad state (identical to harness_implicit.c) */
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

#define NREC (8 * 7 * 6)

static ldouble bpp[NREC][NV], buu[NREC][NV], bdt[NREC];
static int bix[NREC], biy[NREC];

int main(int argc, char **argv)
{
  int rounds = argc > 1 ? atoi(argv[1]) : 30;
  int a, b, s, i, n, r;

  /* same silent-LU setup as production-without-abort (see harness_implicit) */
  gsl_set_error_handler_off();

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  ldouble mindx, mindy, mindz, maxdt;
  set_grid(&mindx, &mindy, &mindz, &maxdt);
  calc_metric();
  alloc_loops();

  n = 0;
  for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 6; s++) {
    int ix = ixs[a], iy = iys[b], iz = 0;
    struct geometry geom;
    fill_geometry(ix, iy, iz, &geom);
    ldouble pp[NV], uu[NV];
    controlled_state(&geom, pp, s % 4);
    ldouble dt = rnd_log(-10, 0);
    p2u(pp, uu, &geom);
    for (i = 0; i < NV; i++) { bpp[n][i] = pp[i]; buu[n][i] = uu[i]; }
    bdt[n] = dt; bix[n] = ix; biy[n] = iy;
    n++;
  }

  double avg = 0., best = 1e300;
  long nok = 0;
  for (r = 0; r <= rounds; r++) { /* round 0 = warmup */
    struct timespec t0, t1;
    long ok = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (n = 0; n < NREC; n++) {
      int ix = bix[n], iy = biy[n], iz = 0;
      for (i = 0; i < NV; i++) {
        set_u(p, i, ix, iy, iz, bpp[n][i]);
        set_u(u, i, ix, iy, iz, buu[n][i]);
      }
      int ret = solve_implicit_lab(ix, iy, iz, bdt[n], 0);
      if (ret == 0) ok++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
    if (r > 0) {
      avg += el;
      if (el < best) best = el;
      nok = ok;
    }
  }
  avg /= (double)rounds;

  fprintf(stderr, "battery: %d cells, %ld converge\n", NREC, nok);
  fprintf(stderr,
          "C     : battery avg %.3f ms, best %.3f ms | %.0f ns/solve (avg over %d rounds)\n",
          avg / 1e6, best / 1e6, avg / (double)NREC, rounds);
  return 0;
}
