/* M8 oracle: opacities + the radiative four-force.
 *
 * Built against koral_lite with PROBLEM=147 (PUFFY): BREMSSTRAHLUNG +
 * SYNCHROTRON (no bridge functions -> Terelfactor active), KLEINNISHINA
 * in the PR_KAPPAES hook, COMPTONIZATION on, HFRAC=1, MASS=10,
 * single-temperature gas (Te = Ti = Tgas), no photon number
 * (Trad = TradBB).
 *
 * States are controlled sweeps over (rho, Tgas, zeta=Trad/Tgas, plasma
 * beta, velocities): temperatures 1e5..1e12, densities 1e-12..1e-4 GU,
 * LTE-comoving records included so the Gi cancellation is exercised.
 *
 * Emitted KGLD files (in/out counts in parentheses):
 *   rad_thermo.kgld (36/13)  geom23 pp13 -> Tgas Te ne Ehat TradBB
 *                            kappaes_state kGasAbs kRadAbs kGasNum
 *                            kRadNum kGasRoss kRadRoss kappa
 *   rad_opac.kgld   (36/3)   geom23 pp13 -> kappa kappaes chi
 *                            (standalone pp-level entry points;
 *                             calc_kappaes uses Trad = Te here)
 *   rad_gi.kgld     (36/8)   geom23 pp13 -> Gi_lab4 Gi_ff4
 *                            (calc_Gi types 1 and 0; thermal == total)
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

/* deterministic xorshift64*, distinct seed */
static uint64_t rngstate = 0x4b4f52414c4f5041ULL; /* "KORALOPA" */
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

static int put_geom23(double *out, struct geometry *g)
{
  int n = put_sym10(out, g->gg);
  n += put_sym10(out + n, g->GG);
  out[n++] = g->gdet;
  out[n++] = g->alpha;
  out[n++] = g->gttpert;
  return n;
}

/* Controlled gas+rad state: Tgas, zeta = Trad_target/Tgas, plasma beta.
 * variant 0: generic; 1: LTE comoving (zeta=1, urad=ugas); 2: B=0;
 * 3: cold+hot extremes */
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

  /* magnetic field with plasma beta 1e-2..1e2 (bsq roughly targeted) */
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

/* interior PUFFY cells, r > 2.6 (same battery as the other harnesses) */
static const int ixs[8] = {24, 48, 96, 144, 192, 240, 300, 370};
static const int iys[7] = {2, 45, 90, 179, 270, 320, 357};

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";
  int a, b, s, i;

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  ldouble mindx, mindy, mindz, maxdt;
  set_grid(&mindx, &mindy, &mindz, &maxdt);
  calc_metric();

  /* ---- A: fill_struct_of_state (rad subset) --------------------------- */
  {
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "rad_thermo.kgld", nrec, 36, 13);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      controlled_state(&geom, pp, s);

      struct struct_of_state state;
      fill_struct_of_state(pp, &geom, &state);

      double in[36], out[13];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      n = 0;
      out[n++] = state.Tgas;
      out[n++] = state.Te;
      out[n++] = state.ne;
      out[n++] = state.Ehat;
      out[n++] = state.TradBB;
      out[n++] = state.kappaes;
      out[n++] = state.opac.kappaGasAbs;
      out[n++] = state.opac.kappaRadAbs;
      out[n++] = state.opac.kappaGasNum;
      out[n++] = state.opac.kappaRadNum;
      out[n++] = state.opac.kappaGasRoss;
      out[n++] = state.opac.kappaRadRoss;
      out[n++] = state.opac.kappaGasAbs; /* kappa == kappaGasAbs here */
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 13, f);
    }
    fclose(f);
  }

  /* ---- B: standalone pp-level opacities (calc_chi path) ---------------- */
  {
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "rad_opac.kgld", nrec, 36, 3);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      controlled_state(&geom, pp, s);

      struct opacities opac;
      ldouble kappa = calc_kappa(pp, &geom, &opac);
      ldouble kappaes = calc_kappaes(pp, &geom); /* Trad = Te flavor */
      ldouble chi = calc_chi(pp, &geom);

      double in[36], out[3];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      out[0] = kappa;
      out[1] = kappaes;
      out[2] = chi;
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 3, f);
    }
    fclose(f);
  }

  /* ---- C: calc_Gi (thermal four-force incl. Comptonization) ----------- */
  {
    uint64_t nrec = 8 * 7 * 6;
    FILE *f = open_kgld(outdir, "rad_gi.kgld", nrec, 36, 8);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 6; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      controlled_state(&geom, pp, s % 4); /* extra generic + LTE rounds */

      ldouble gi_lab[4], gi_ff[4];
      calc_Gi(pp, &geom, gi_lab, 0.0, 1, 0);
      calc_Gi(pp, &geom, gi_ff, 0.0, 0, 0);

      double in[36], out[8];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      for (i = 0; i < 4; i++) out[i] = gi_lab[i];
      for (i = 0; i < 4; i++) out[4 + i] = gi_ff[i];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 8, f);
    }
    fclose(f);
  }

  fprintf(stderr, "harness_opac: done\n");
  return 0;
}
