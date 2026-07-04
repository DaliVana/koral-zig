/* M7 oracle: M1 radiation basics — calc_Rij_M1 + calc_ff_Rtt, the
 * closed-form u2p_rad inversion (incl. its cold fallback branches), the
 * τ-limited radiative wavespeeds, and check_floors_rad.
 *
 * Built against koral_lite with PROBLEM=147 (PUFFY): RADIATION on,
 * VELPRIMRAD=VELR, GDETIN=1, GAMMAMAXRAD=10, EERHORATIO 1e-20/1e4,
 * EEUURATIO 1e-20/1e20, default τ-damping rv2 = (4/3)²/τ², TNZ==1 (the
 * wavespeed core leaves z slots untouched — only x/y are recorded).
 *
 * Every record carries the geometry the C code actually used (see
 * harness_state.c). Emitted KGLD files (in/out counts in parentheses):
 *   rad_rij.kgld        (33/21)  gg10 GG10 pp13 -> Rij16 Rtt ucon4
 *   rad_u2prad.kgld     (31/6)   geom23 uuEE..FZ4 guess4 -> ret cor ppEE..FZ4
 *   rad_wavespeeds.kgld (36/8)   gg10 GG10 pp13 tau3 -> axl0 axr0 ayl0 ayr0
 *                                                       axl axr ayl ayr
 *   rad_floors.kgld     (36/14)  geom23 pp13 -> ret pp13
 */

#include "ko.h"
#include <stdint.h>

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

/* deterministic xorshift64*, distinct seed from the other harnesses */
static uint64_t rngstate = 0x4b4f52414c524144ULL; /* "KORALRAD" */
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

/* VELR components with exact Lorentz factor gamma wrt the normal observer */
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

static int put_geom20(double *out, struct geometry *g)
{
  int n = put_sym10(out, g->gg);
  n += put_sym10(out + n, g->GG);
  return n;
}

static int put_geom23(double *out, struct geometry *g)
{
  int n = put_geom20(out, g);
  out[n++] = g->gdet;
  out[n++] = g->alpha;
  out[n++] = g->gttpert;
  return n;
}

/* random full PUFFY primitive state (gas + radiation) */
static void random_state(struct geometry *geom, ldouble *pp,
                         double max_gamma, double max_gamma_rad)
{
  double v[3];
  double rho = rnd_log(-10, 2);
  double uint_ = rho * rnd_log(-6, 2);
  pp[RHO] = rho;
  pp[UU] = uint_;
  velr_gamma(geom->gg, 1. + (max_gamma - 1.) * rnd(), v);
  pp[VX] = v[0]; pp[VY] = v[1]; pp[VZ] = v[2];
  pp[ENTR] = calc_Sfromu(rho, uint_, geom->ix, geom->iy, geom->iz);
  pp[B1] = pp[B2] = pp[B3] = 0.;
  pp[EE] = rnd_log(-20, 1);
  velr_gamma(geom->gg, 1. + (max_gamma_rad - 1.) * rnd(), v);
  pp[FX] = v[0]; pp[FY] = v[1]; pp[FZ] = v[2];
}

/* interior PUFFY cells, r > 2.6 (same battery as harness_state.c) */
static const int ixs[8] = {24, 48, 96, 144, 192, 240, 300, 370};
static const int iys[7] = {2, 45, 90, 179, 270, 320, 357};

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";
  int a, b, s, i, j;

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  ldouble mindx, mindy, mindz, maxdt;
  set_grid(&mindx, &mindy, &mindz, &maxdt);
  calc_metric();

  /* ---- A: calc_Rij_M1 + calc_ff_Rtt ---------------------------------- */
  {
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "rad_rij.kgld", nrec, 33, 21);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      random_state(&geom, pp, 4.0, 8.0);

      ldouble Rij[4][4], Rtt, ucon[4];
      calc_Rij_M1(pp, &geom, Rij);
      calc_ff_Rtt(pp, &Rtt, ucon, &geom);

      double in[33], out[21];
      int n = put_geom20(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      n = 0;
      for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = Rij[i][j];
      out[n++] = Rtt;
      for (i = 0; i < 4; i++) out[n++] = ucon[i];
      fwrite(in, 8, 33, f);
      fwrite(out, 8, 21, f);
    }
    fclose(f);
  }

  /* ---- B: u2p_rad (closed form + cold fallbacks) ---------------------- */
  {
    /* 6 variants per (cell, sample):
       0: round-trip, gamma_rad < 8 (nonfailure branch)
       1: round-trip, gamma_rad in (15, 60) -> exceeds GAMMAMAXRAD=10 (cold fast)
       2: F = 0 exactly (gamma = 1, the gamma2b cancellation branch)
       3: uu[EE] sign-flipped (Erf < 0 -> cold)
       4: uu[EE] scaled by 1e-70 (Erf < ERADFLOOR -> cold)
       5: random garbage R^t_mu of mixed magnitudes                     */
    uint64_t nrec = 8 * 7 * 6;
    FILE *f = open_kgld(outdir, "rad_u2prad.kgld", nrec, 31, 6);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 6; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV], uu[NV];
      double grad = 1. + 7. * rnd();
      if (s == 1) grad = 15. + 45. * rnd();
      random_state(&geom, pp, 4.0, grad);
      if (s == 2) { pp[FX] = pp[FY] = pp[FZ] = 0.; }
      p2u(pp, uu, &geom);
      if (s == 3) uu[EE] = -uu[EE];
      if (s == 4) { uu[EE] *= 1e-70; uu[FX] *= 1e-70; uu[FY] *= 1e-70; uu[FZ] *= 1e-70; }
      if (s == 5) {
        uu[EE] = rnd_pm() * rnd_log(-15, 5) * geom.gdet;
        uu[FX] = rnd_pm() * rnd_log(-15, 5) * geom.gdet;
        uu[FY] = rnd_pm() * rnd_log(-15, 5) * geom.gdet;
        uu[FZ] = rnd_pm() * rnd_log(-15, 5) * geom.gdet;
      }

      /* guess = some other state; the inversion must not read it, and on
         ret == -1 it must pass through unchanged */
      ldouble guess[4] = {0.37, 0.11, -0.05, 0.02};
      pp[EE] = guess[0]; pp[FX] = guess[1]; pp[FY] = guess[2]; pp[FZ] = guess[3];

      int cor = 0;
      int ret = u2p_rad(uu, pp, &geom, &cor);

      double in[31], out[6];
      int n = put_geom23(in, &geom);
      in[n++] = uu[EE]; in[n++] = uu[FX]; in[n++] = uu[FY]; in[n++] = uu[FZ];
      for (i = 0; i < 4; i++) in[n++] = guess[i];
      out[0] = (double)ret;
      out[1] = (double)cor;
      out[2] = pp[EE]; out[3] = pp[FX]; out[4] = pp[FY]; out[5] = pp[FZ];
      fwrite(in, 8, 31, f);
      fwrite(out, 8, 6, f);
    }
    fclose(f);
  }

  /* ---- C: calc_rad_wavespeeds (τ-limiter) ------------------------------ */
  {
    /* tau variants: 0 (thin branch), moderate, thick, mixed per-dim */
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "rad_wavespeeds.kgld", nrec, 36, 8);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      random_state(&geom, pp, 4.0, 8.0);

      ldouble tautot[3] = {0., 0., 0.};
      if (s == 1) { tautot[0] = rnd_log(-2, 1); tautot[1] = rnd_log(-2, 1); tautot[2] = rnd_log(-2, 1); }
      if (s == 2) { tautot[0] = rnd_log(2, 6); tautot[1] = rnd_log(2, 6); tautot[2] = rnd_log(2, 6); }
      if (s == 3) { tautot[0] = 0.; tautot[1] = rnd_log(-1, 4); tautot[2] = rnd_log(-1, 4); }

      ldouble aval[18];
      for (i = 0; i < 18; i++) aval[i] = 0.;
      calc_rad_wavespeeds(pp, &geom, tautot, aval, 0);

      double in[36], out[8];
      int n = put_geom20(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      for (i = 0; i < 3; i++) in[n++] = tautot[i];
      /* TNZ==1: z slots never written by the core — record x/y only */
      out[0] = aval[0]; out[1] = aval[1]; out[2] = aval[2]; out[3] = aval[3];
      out[4] = aval[6]; out[5] = aval[7]; out[6] = aval[8]; out[7] = aval[9];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 8, f);
    }
    fclose(f);
  }

  /* ---- D: check_floors_rad -------------------------------------------- */
  {
    /* biased so every floor fires somewhere: EE down to 1e-90 hits
       ERADFLOOR; EE up to 10 over rho down to 1e-10 hits EERHORATIOMAX */
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "rad_floors.kgld", nrec, 36, 14);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      random_state(&geom, pp, 3.0, 4.0);
      if (s == 1) pp[EE] = rnd_log(-90, -75);
      if (s == 2) { pp[EE] = rnd_log(-1, 1); pp[RHO] = rnd_log(-10, -5); pp[UU] = pp[RHO] * rnd_log(-2, 0); }
      if (s == 3) pp[EE] = pp[RHO] * rnd_log(-26, -18);

      double in[36], out[14];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];

      int ret = check_floors_rad(pp, VELPRIMRAD, &geom);

      out[0] = (double)ret;
      for (i = 0; i < NV; i++) out[1 + i] = pp[i];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 14, f);
    }
    fclose(f);
  }

  fprintf(stderr, "harness_rad: done\n");
  return 0;
}
