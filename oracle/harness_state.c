/* M2/M3 oracle: velocity conversions, Lorentz boosts, primitive coordinate
 * transforms, p2u, u2p_solver (hot + entropy), check_floors_mhd.
 *
 * Built against koral_lite with PROBLEM=147 (PUFFY): NV=13, VELPRIM=VELR,
 * GDETIN=1, U2P_SOLVER_W + U2P_EQS_NOBLE, B2RHOFLOORFRAME=DRIFTFRAME.
 *
 * Every record carries the geometry the C code actually used (gg, GG and
 * where needed gdet, alpha, gttpert), so the Zig side replays pure state
 * algebra with identical inputs — the MKS2 two-pi metric spread never
 * enters these comparisons. Symmetric 4x4 blocks are stored as 10
 * upper-triangle values (00,01,02,03,11,12,13,22,23,33).
 *
 * Emitted KGLD files (in/out counts in parentheses):
 *   relele_convvels.kgld  (25/4)   gg10 GG10 w1 w2 v3 -> u2[4]
 *   relele_boost.kgld     (27/24)  gg10 GG10 vel3 A4 -> L16 lab2ff4 ff2lab4
 *   frames_transpall.kgld (58/13)  co1 co2 x3 pp13 g1(gg10 GG10) g2(gg10 GG10) -> pp2[13]
 *   physics_tij.kgld      (33/33)  gg10 GG10 pp13 -> T16 ucon4 ucov4 bcon4 bcov4 bsq
 *   p2u.kgld              (36/13)  gg10 GG10 gdet alpha gttpert pp13 -> uu13
 *   u2p_solver.kgld       (49/20)  geom23 uu13 guess13 -> ret_hot pp_hot9 ret_entr pp_entr9
 *   floors.kgld           (36/14)  geom23 pp13 -> ret pp13
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

/* deterministic xorshift64* */
static uint64_t rngstate = 0x4b4f52414c5a4947ULL; /* "KORALZIG" */
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

/* random full PUFFY primitive state; gamma/b2orho caps parameterized */
static void random_state(struct geometry *geom, ldouble *pp,
                         double max_gamma, double max_b2orho)
{
  int i;
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
  for (i = 0; i < NV; i++)
    if (!isfinite((double)pp[i])) { fprintf(stderr, "bad state\n"); exit(1); }
}

/* cell battery: interior PUFFY cells with r > 2.6 at the low end so BL
 * geometries (horizon at r=2) stay regular */
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

  /* ---- A: conv_vels, all 9 (which1,which2) pairs -------------------- */
  {
    const int wpairs[9][2] = {{VEL4,VEL4},{VEL3,VEL3},{VELR,VELR},
                              {VEL4,VEL3},{VEL3,VEL4},{VEL3,VELR},
                              {VEL4,VELR},{VELR,VEL4},{VELR,VEL3}};
    uint64_t nrec = 8 * 3 * 9;
    FILE *f = open_kgld(outdir, "relele_convvels.kgld", nrec, 25, 4);
    for (a = 0; a < 8; a++) for (b = 0; b < 3; b++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[2 * b + 1], 0, &geom);
      double vr[3];
      velr_gamma(geom.gg, 1.05 + 7. * rnd(), vr);
      ldouble base[4] = {0., vr[0], vr[1], vr[2]};
      for (s = 0; s < 9; s++) {
        int w1 = wpairs[s][0], w2 = wpairs[s][1];
        ldouble u1[4], u2[4];
        conv_vels(base, u1, VELR, w1, geom.gg, geom.GG);
        u1[0] = 0.; /* spatial-only input, like primitives */
        conv_vels(u1, u2, w1, w2, geom.gg, geom.GG);
        double in[25], out[4];
        int n = put_geom20(in, &geom);
        in[n++] = (double)w1;
        in[n++] = (double)w2;
        in[n++] = u1[1]; in[n++] = u1[2]; in[n++] = u1[3];
        for (i = 0; i < 4; i++) out[i] = u2[i];
        fwrite(in, 8, 25, f);
        fwrite(out, 8, 4, f);
      }
    }
    fclose(f);
  }

  /* ---- B: Lorentz matrix + vector boosts ---------------------------- */
  {
    uint64_t nrec = 8 * 3 * 3;
    FILE *f = open_kgld(outdir, "relele_boost.kgld", nrec, 27, 24);
    for (a = 0; a < 8; a++) for (b = 0; b < 3; b++) for (s = 0; s < 3; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[2 * b], 0, &geom);
      ldouble pp[NV];
      random_state(&geom, pp, 3.0, 0.);
      ldouble A[4], L[4][4], A2[4], A3[4];
      for (i = 0; i < 4; i++) A[i] = 2. * rnd_pm();
      calc_Lorentz_lab2ff(pp, geom.gg, geom.GG, L);
      boost2_lab2ff(A, A2, pp, geom.gg, geom.GG);
      boost2_ff2lab(A, A3, pp, geom.gg, geom.GG);
      double in[27], out[24];
      int n = put_geom20(in, &geom);
      in[n++] = pp[VX]; in[n++] = pp[VY]; in[n++] = pp[VZ];
      for (i = 0; i < 4; i++) in[n++] = A[i];
      n = 0;
      for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = L[i][j];
      for (i = 0; i < 4; i++) out[n++] = A2[i];
      for (i = 0; i < 4; i++) out[n++] = A3[i];
      fwrite(in, 8, 27, f);
      fwrite(out, 8, 24, f);
    }
    fclose(f);
  }

  /* ---- C: trans_pall_coco, MKS2 <-> KS/BL ---------------------------- */
  {
    const int co2s[2] = {KSCOORDS, BLCOORDS};
    uint64_t nrec = 8 * 3 * 2 * 2;
    FILE *f = open_kgld(outdir, "frames_transpall.kgld", nrec, 58, 13);
    for (a = 0; a < 8; a++) for (b = 0; b < 3; b++) for (s = 0; s < 2; s++) {
      int ix = ixs[a], iy = iys[2 * b + 1];
      struct geometry gmy, garb;
      fill_geometry(ix, iy, 0, &gmy);
      fill_geometry_arb(ix, iy, 0, &garb, co2s[s]);
      int dir;
      for (dir = 0; dir < 2; dir++) {
        struct geometry *g1 = dir == 0 ? &gmy : &garb;
        struct geometry *g2 = dir == 0 ? &garb : &gmy;
        int co1 = dir == 0 ? MYCOORDS : co2s[s];
        int co2 = dir == 0 ? co2s[s] : MYCOORDS;
        ldouble pp[NV], pp2[NV];
        random_state(g1, pp, 3.0, 10.0);
        /* MUST be called aliased: trans_prad_coco's init loop copies ppin
         * over ALL of ppout, clobbering trans_pmhd_coco's result when the
         * arrays are distinct (frames.c:154). KORAL always aliases. */
        for (i = 0; i < NV; i++) pp2[i] = pp[i];
        trans_pall_coco(pp2, pp2, co1, co2, g1->xxvec, g1, g2);
        double in[58], out[13];
        int n = 0;
        in[n++] = (double)co1;
        in[n++] = (double)co2;
        in[n++] = g1->xxvec[1]; in[n++] = g1->xxvec[2]; in[n++] = g1->xxvec[3];
        for (i = 0; i < NV; i++) in[n++] = pp[i];
        n += put_geom20(in + n, g1);
        n += put_geom20(in + n, g2);
        for (i = 0; i < NV; i++) out[i] = pp2[i];
        fwrite(in, 8, 58, f);
        fwrite(out, 8, 13, f);
      }
    }
    fclose(f);
  }

  /* ---- D: calc_Tij + four-vectors ------------------------------------ */
  {
    uint64_t nrec = 8 * 7;
    FILE *f = open_kgld(outdir, "physics_tij.kgld", nrec, 33, 33);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      random_state(&geom, pp, 5.0, 50.0);
      ldouble T[4][4], ucon[4], ucov[4], bcon[4], bcov[4], bsq;
      calc_Tij(pp, &geom, T);
      calc_ucon_ucov_from_prims(pp, &geom, ucon, ucov);
      calc_bcon_bcov_bsq_from_4vel(pp, ucon, ucov, &geom, bcon, bcov, &bsq);
      double in[33], out[33];
      int n = put_geom20(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      n = 0;
      for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = T[i][j];
      for (i = 0; i < 4; i++) out[n++] = ucon[i];
      for (i = 0; i < 4; i++) out[n++] = ucov[i];
      for (i = 0; i < 4; i++) out[n++] = bcon[i];
      for (i = 0; i < 4; i++) out[n++] = bcov[i];
      out[n++] = bsq;
      fwrite(in, 8, 33, f);
      fwrite(out, 8, 33, f);
    }
    fclose(f);
  }

  /* ---- E: p2u (mhd + rad) -------------------------------------------- */
  {
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "p2u.kgld", nrec, 36, 13);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV], uu[NV];
      random_state(&geom, pp, 5.0, 50.0);
      if (s == 1) { pp[VX] = pp[VY] = pp[VZ] = 0.; }           /* at rest */
      if (s == 2) { pp[B1] = pp[B2] = pp[B3] = 0.; }           /* unmagnetized */
      if (s == 3) {                                            /* corners */
        double v[3];
        pp[RHO] = 1e-25;
        pp[UU] = pp[RHO] * 1e-2;
        pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], geom.ix, geom.iy, geom.iz);
        velr_gamma(geom.gg, 9.9, v);
        pp[VX] = v[0]; pp[VY] = v[1]; pp[VZ] = v[2];
        /* keep b2/rho at the floor ceiling — beyond it double precision
         * genuinely loses ~8 digits in the cancelling momentum rows and
         * C/Zig FMA differences dominate the comparison */
        velr_gamma(geom.gg, sqrt(1. + 50. * pp[RHO]), v);
        pp[B1] = v[0]; pp[B2] = v[1]; pp[B3] = v[2];
      }
      p2u(pp, uu, &geom);
      double in[36], out[13];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      for (i = 0; i < NV; i++) out[i] = uu[i];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 13, f);
    }
    fclose(f);
  }

  /* ---- F: u2p_solver_W, hot + entropy, independent from same guess ---- */
  {
    uint64_t nrec = 8 * 7 * 4;
    FILE *f = open_kgld(outdir, "u2p_solver.kgld", nrec, 49, 20);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 4; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp0[NV], uu[NV], guess[NV], pph[NV], ppe[NV];
      if (s == 0) random_state(&geom, pp0, 3.0, 10.0);
      if (s == 1) { random_state(&geom, pp0, 9.5, 0.0); }       /* fast, B=0 */
      if (s == 2) { random_state(&geom, pp0, 2.0, 50.0); }      /* magnetized */
      if (s == 3) { random_state(&geom, pp0, 2.0, 5.0); }
      p2u(pp0, uu, &geom);
      for (i = 0; i < NV; i++) guess[i] = pp0[i];
      if (s == 0) { guess[RHO] *= 1.1; guess[UU] *= 0.9; }
      if (s == 1) { guess[RHO] *= 100.; }                      /* wild guess */
      if (s == 2) { guess[RHO] *= 0.01; guess[UU] *= 10.; }
      if (s == 3) { uu[UU] = -uu[UU]; }                        /* break energy */
      for (i = 0; i < NV; i++) { pph[i] = guess[i]; ppe[i] = guess[i]; }
      int rh = u2p_solver_mhd(uu, pph, &geom, U2P_HOT, 0);
      int re = u2p_solver_mhd(uu, ppe, &geom, U2P_ENTROPY, 0);
      double in[49], out[20];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = uu[i];
      for (i = 0; i < NV; i++) in[n++] = guess[i];
      n = 0;
      out[n++] = (double)rh;
      for (i = 0; i < NVMHD; i++) out[n++] = pph[i];
      out[n++] = (double)re;
      for (i = 0; i < NVMHD; i++) out[n++] = ppe[i];
      fwrite(in, 8, 49, f);
      fwrite(out, 8, 20, f);
    }
    fclose(f);
  }

  /* ---- G: check_floors_mhd ------------------------------------------- */
  {
    uint64_t nrec = 8 * 7 * 3;
    FILE *f = open_kgld(outdir, "floors.kgld", nrec, 36, 14);
    for (a = 0; a < 8; a++) for (b = 0; b < 7; b++) for (s = 0; s < 3; s++) {
      struct geometry geom;
      fill_geometry(ixs[a], iys[b], 0, &geom);
      ldouble pp[NV];
      if (s == 0) {                     /* rho / uint-ratio floors */
        random_state(&geom, pp, 3.0, 1.0);
        if (rnd() < 0.4) pp[RHO] = 1e-35;
        if (rnd() < 0.4) pp[UU] = pp[RHO] * 1e-14;
        if (rnd() < 0.4) pp[UU] = pp[RHO] * 1e4;
      }
      if (s == 1) {                     /* magnetic floor, drift frame */
        random_state(&geom, pp, 2.5, 1.0);
        double v[3];
        velr_gamma(geom.gg, sqrt(1. + 400. * pp[RHO]), v);
        pp[B1] = v[0]; pp[B2] = v[1]; pp[B3] = v[2];
      }
      if (s == 2) {                     /* gamma ceiling, weak field */
        random_state(&geom, pp, 12.0, 1e-6);
      }
      double in[36], out[14];
      int n = put_geom23(in, &geom);
      for (i = 0; i < NV; i++) in[n++] = pp[i];
      int ret = check_floors_mhd(pp, VELPRIM, &geom);
      n = 0;
      out[n++] = (double)ret;
      for (i = 0; i < NV; i++) out[n++] = pp[i];
      fwrite(in, 8, 36, f);
      fwrite(out, 8, 14, f);
    }
    fclose(f);
  }

  printf("harness_state: golden files written to %s\n", outdir);
  return 0;
}
