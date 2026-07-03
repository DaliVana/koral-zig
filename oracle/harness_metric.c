/* M1 oracle: dump KORAL's metric layer at deterministic points.
 *
 * Built against koral_lite sources with PROBLEM=147 (PUFFY: BHSPIN=0,
 * MKS2 with MKSR0=0.1, MKSH0=0.9, GDETIN=1 => MODYFIKUJKRZYSIE=1).
 *
 * Emits three KGLD files (little-endian: "KGLD", u32 version, u64 nrec,
 * u32 n_in, u32 n_out, then nrec * (n_in+n_out) f64):
 *   metric_points.kgld  in:  coords, x0..x3                       (5)
 *                       out: gg(16) GG(16) gdet dlgdet(3) Kr(64)
 *                            gttpert                              (101)
 *   coco_dxdx.kgld      in:  co1, co2, x0..x3                     (6)
 *                       out: xout(4) dxdx(16)                     (20)
 *   krzysie_grid.kgld   in:  ix, iy                               (2)
 *                       out: x1, x2, Kr_corrected(64)             (66)
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

/* ---- part A: coordinate-pure metric quantities ---- */

static const double ks_r[]   = {1.55, 1.85, 2.0, 2.05, 2.5, 5.0, 21.7, 103.0, 450.0};
static const double bl_r[]   = {2.05, 2.5, 5.0, 21.7, 103.0, 450.0};
static const double mks2_r[] = {1.85, 2.0, 2.6, 5.0, 21.7, 103.0, 450.0};
static const double ths[]    = {0.005, 0.3, 1.5707963267948966, 2.7, 3.136592653589793};
static const double x2s[]    = {0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999};
static const double mink_p[][3] = {{-3.4, 0.2, 5.7}, {0.2, 5.7, -3.4}, {5.7, -3.4, 0.2}};

#define NREC_A (sizeof(ks_r)/sizeof(double)*5 + sizeof(bl_r)/sizeof(double)*5 \
              + sizeof(mks2_r)/sizeof(double)*7 + 3)

static void metric_record(FILE *f, int coords, double x1, double x2, double x3)
{
  ldouble xx[4] = {0., x1, x2, x3};
  double in[5] = {(double)coords, xx[0], xx[1], xx[2], xx[3]};
  double out[101];
  ldouble g[4][5], G[4][5], Kr[4][4][4];
  int i, j, k, n = 0;

  calc_g_arb(xx, g, coords);
  calc_G_arb(xx, G, coords);
  for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = g[i][j];
  for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = G[i][j];
  out[n++] = calc_gdet_arb(xx, coords);
  for (i = 0; i < 3; i++) out[n++] = calc_dlgdet_arb(xx, i, coords);
  calc_Krzysie_arb(xx, Kr, coords);
  for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) for (k = 0; k < 4; k++)
    out[n++] = Kr[i][j][k];
  out[n++] = calc_gttpert_arb(xx, coords);

  fwrite(in, 8, 5, f);
  fwrite(out, 8, 101, f);
}

/* ---- part B: coco + Jacobians ---- */

static void coco_record(FILE *f, int co1, int co2, double x1, double x2, double x3)
{
  ldouble xx[4] = {0., x1, x2, x3}, xout[4];
  ldouble dd[4][4];
  double in[6] = {(double)co1, (double)co2, 0., x1, x2, x3};
  double out[20];
  int i, j, n = 0;

  coco_N(xx, xout, co1, co2);
  calc_dxdx_arb(xx, dd, co1, co2);
  for (i = 0; i < 4; i++) out[n++] = xout[i];
  for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) out[n++] = dd[i][j];

  fwrite(in, 8, 6, f);
  fwrite(out, 8, 20, f);
}

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";
  int i, j;

  global_time = 0.;

  /* A: pure points */
  {
    FILE *f = open_kgld(outdir, "metric_points.kgld", NREC_A, 5, 101);
    for (i = 0; i < (int)(sizeof(ks_r)/sizeof(double)); i++)
      for (j = 0; j < 5; j++)
        metric_record(f, KSCOORDS, ks_r[i], ths[j], 0.3);
    for (i = 0; i < (int)(sizeof(bl_r)/sizeof(double)); i++)
      for (j = 0; j < 5; j++)
        metric_record(f, KERRCOORDS, bl_r[i], ths[j], 0.3);
    for (i = 0; i < (int)(sizeof(mks2_r)/sizeof(double)); i++)
      for (j = 0; j < 7; j++)
        metric_record(f, MKS2COORDS, log(mks2_r[i] - MKSR0), x2s[j], 0.3);
    for (i = 0; i < 3; i++)
      metric_record(f, MINKCOORDS, mink_p[i][0], mink_p[i][1], mink_p[i][2]);
    fclose(f);
  }

  /* B: transforms (radii outside the horizon so the BL t-shift is defined) */
  {
    const double rs[] = {2.05, 2.5, 5.0, 21.7, 103.0, 450.0};
    const double th3[] = {0.3, 1.2, 2.7};
    const double xx2[] = {0.1, 0.5, 0.9};
    const int nr = 6, na = 3;
    uint64_t nrec = (uint64_t)(6 * nr * na);
    FILE *f = open_kgld(outdir, "coco_dxdx.kgld", nrec, 6, 20);
    for (i = 0; i < nr; i++)
      for (j = 0; j < na; j++) {
        coco_record(f, KSCOORDS, MKS2COORDS, rs[i], th3[j], 0.3);
        coco_record(f, KSCOORDS, KERRCOORDS, rs[i], th3[j], 0.3);
        coco_record(f, KERRCOORDS, KSCOORDS, rs[i], th3[j], 0.3);
        coco_record(f, MKS2COORDS, KSCOORDS, log(rs[i] - MKSR0), xx2[j], 0.3);
        coco_record(f, MKS2COORDS, KERRCOORDS, log(rs[i] - MKSR0), xx2[j], 0.3);
        coco_record(f, KERRCOORDS, MKS2COORDS, rs[i], th3[j], 0.3);
      }
    fclose(f);
  }

  /* C: grid-corrected Christoffels (MODYFIKUJKRZYSIE) on the PUFFY grid */
  {
    init_pointers();
    initialize_arrays();
    initialize_constants();
    ldouble mindx, mindy, mindz, maxdt;
    set_grid(&mindx, &mindy, &mindz, &maxdt);

    uint64_t nrec = 8 * 8;
    FILE *f = open_kgld(outdir, "krzysie_grid.kgld", nrec, 2, 66);
    int ixs[8] = {0, 48, 96, 144, 192, 240, 300, 383};
    int iys[8] = {0, 45, 90, 135, 180, 225, 300, 359};
    int a, b, ii, jj, kk;
    for (a = 0; a < 8; a++)
      for (b = 0; b < 8; b++) {
        ldouble Kr[4][4][4];
        calc_Krzysie_at_center(ixs[a], iys[b], 0, Kr);
        double in[2] = {(double)ixs[a], (double)iys[b]};
        double out[66];
        int n = 0;
        out[n++] = get_x(ixs[a], 0);
        out[n++] = get_x(iys[b], 1);
        for (ii = 0; ii < 4; ii++) for (jj = 0; jj < 4; jj++) for (kk = 0; kk < 4; kk++)
          out[n++] = Kr[ii][jj][kk];
        fwrite(in, 8, 2, f);
        fwrite(out, 8, 66, f);
      }
    fclose(f);
  }

  printf("harness_metric: golden files written to %s\n", outdir);
  return 0;
}
