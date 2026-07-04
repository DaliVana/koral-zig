/* M4 oracle: avg2point reconstruction, gas wavespeeds, f_flux_prime.
 *
 * Built against koral_lite with PROBLEM=147 (PUFFY): INT_ORDER=2 (PPM),
 * FLUXLIMITER=0, MINMOD_THETA=1.5, LAXF, RADVISCOSITY=SHEARVISCOSITY.
 * Rijviscglobal is explicitly zeroed so the radiative flux rows are the
 * pure M1 tensor (the viscous correction is M12 physics and is zero at
 * t=0; the Zig side matches this).
 *
 * Emitted KGLD files:
 *   recon.kgld      (12/2)  u5 dx5 param theta -> ul ur
 *   wavespeed.kgld  (36/4)  geom23 pp13 -> ahd{xl,xr,yl,yr}
 *                           (z is uninitialized in C for TNZ==1)
 *   fluxprime.kgld  (37/13) geom23(face) idim pp13 -> ff13
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

static uint64_t rngstate = 0x4d34464c55584f4bULL;
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

static void random_state(struct geometry *geom, ldouble *pp,
                         double max_gamma, double max_b2orho)
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
  if (max_b2orho > 0. && rnd() < 0.8) {
    double f = rnd_log(-8, log10(max_b2orho));
    velr_gamma(geom->gg, sqrt(1. + f * rho), v);
    pp[B1] = v[0]; pp[B2] = v[1]; pp[B3] = v[2];
  }
  pp[EE] = rnd_log(-20, 1);
  velr_gamma(geom->gg, 1. + 2. * rnd(), v);
  pp[FX] = v[0]; pp[FY] = v[1]; pp[FZ] = v[2];
}

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

  /* zero the viscous radiation tensor everywhere we may touch it */
  {
    int ix, iy, ii, jj;
    for (ix = -NG; ix < NX + NG; ix++)
      for (iy = -NG; iy < NY + NG; iy++)
        for (ii = 0; ii < 4; ii++)
          for (jj = 0; jj < 4; jj++)
            set_Tfull(Rijviscglobal, ii, jj, ix, iy, 0, 0.);
  }

  /* ---- A: avg2point ---------------------------------------------------- */
  {
    const int ncall = 24;
    uint64_t nrec = (uint64_t)ncall * NV;
    FILE *f = open_kgld(outdir, "recon.kgld", nrec, 12, 2);
    int c;
    for (c = 0; c < ncall; c++) {
      ldouble um2[NV], um1[NV], u0[NV], up1[NV], up2[NV], ul[NV], ur[NV];
      ldouble dx[5];
      int param = c % 3;
      double theta = (c % 2) ? 1.5 : 1.0 + rnd();
      if (c % 2) { for (i = 0; i < 5; i++) dx[i] = 1.; }
      else       { for (i = 0; i < 5; i++) dx[i] = 0.5 + 1.5 * rnd(); }

      for (i = 0; i < NV; i++) {
        double st[5];
        switch (i % 5) {
          case 0: { /* smooth quadratic + noise */
            double q = rnd_pm();
            for (j = 0; j < 5; j++) st[j] = q * (j - 2.) * (j - 2.) + rnd_pm() * 0.01;
            break;
          }
          case 1: /* monotone ramp */
            st[0] = rnd_pm();
            for (j = 1; j < 5; j++) st[j] = st[j - 1] + rnd();
            break;
          case 2: /* step */
            for (j = 0; j < 5; j++) st[j] = (j < 2 + (int)(rnd() * 2.)) ? 0. : 1.;
            break;
          case 3: /* extremum */
            for (j = 0; j < 5; j++) st[j] = -(j - 2.) * (j - 2.) * rnd() + rnd_pm() * 0.1;
            break;
          default: /* random */
            for (j = 0; j < 5; j++) st[j] = rnd_pm();
        }
        um2[i] = st[0]; um1[i] = st[1]; u0[i] = st[2]; up1[i] = st[3]; up2[i] = st[4];
      }

      avg2point(um2, um1, u0, up1, up2, ul, ur, dx[0], dx[1], dx[2], dx[3], dx[4],
                param, theta);

      for (i = 0; i < NV; i++) {
        double in[12], out[2];
        in[0] = um2[i]; in[1] = um1[i]; in[2] = u0[i]; in[3] = up1[i]; in[4] = up2[i];
        for (j = 0; j < 5; j++) in[5 + j] = dx[j];
        in[10] = (double)param;
        in[11] = theta;
        out[0] = ul[i];
        out[1] = ur[i];
        fwrite(in, 8, 12, f);
        fwrite(out, 8, 2, f);
      }
    }
    fclose(f);
  }

  /* ---- B: gas wavespeeds ----------------------------------------------- */
  {
    uint64_t nrec = 8 * 7 * 3;
    FILE *f = open_kgld(outdir, "wavespeed.kgld", nrec, 36, 4);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 3; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV], aaa[24];
      random_state(&geom, pp, 5.0, 50.0);
      if (s == 1) { pp[VX] = pp[VY] = pp[VZ] = 0.; }
      if (s == 2) { pp[B1] = pp[B2] = pp[B3] = 0.; }
      calc_wavespeeds_lr_pure(pp, &geom, aaa);
      double in[36], out[4];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      for (i = 0; i < 4; i++) out[i] = aaa[i];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 4, f);
    }
    fclose(f);
  }

  /* ---- C: f_flux_prime at faces ---------------------------------------- */
  {
    uint64_t nrec = 8 * 7 * 2 * 2;
    FILE *f = open_kgld(outdir, "fluxprime.kgld", nrec, 37, 13);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 2; s++) {
      int idim;
      for (idim = 0; idim < 2; idim++) {
        int ix = ixs[a], iy = iys[b];
        struct geometry geom;
        fill_geometry_face(ix, iy, 0, idim, &geom);
        ldouble pp[NV], ff[NV];
        random_state(&geom, pp, 3.0, 10.0);
        if (s == 1 && rnd() < 0.5) { pp[VX] = pp[VY] = pp[VZ] = 0.; }
        f_flux_prime(pp, idim, ix, iy, 0, ff, 0);
        double in[37], out[13];
        int n = put_geom23(in, &geom);
        in[n++] = (double)idim;
        for (i = 0; i < NV; i++) in[n++] = pp[i];
        for (i = 0; i < NV; i++) out[i] = ff[i];
        fwrite(in, 8, 37, f);
        fwrite(out, 8, 13, f);
      }
    }
    fclose(f);
  }

  printf("harness_flux: golden files written to %s\n", outdir);
  return 0;
}
