/* M5/M6 oracle: forced-dt step tests (KSTP) + flux_ct / calc_BfromA goldens.
 *
 * Built three times against koral_lite with PROBLEM = 200 (ZIGSOD),
 * 201 (ZIGOT) or 202 (ZIGMHDTUBE). Initializes exactly like ko.c
 * (set_initial_profile → set_bc → [VECPOTGIVEN: calc_BfromA → set_bc]),
 * replicates problem.c's solve preamble (max_ws=10⁴ dt guess) and then the
 * RK2IMEX stage block verbatim (problem.c:141-402, serial path), dumping
 * the full domain state each step.
 *
 * KSTP format (little-endian):
 *   "KSTP" u32 version=1, u32 nx,ny,nz,nv, u64 nrec(=nsteps+1)
 *   per record: f64 t, f64 dt, u[nx*ny*nz*nv], p[same], flags[nf*ncell]
 *   (iv fastest, then ix, iy, iz; flags = ENTROPYFLAG, HDFIXUPFLAG — plus
 *    RADFIXUPFLAG, RADIMPFIXUPFLAG for RADIATION builds (nf = 4, else 2) —
 *    as f64; record 0 is the post-init state with dt=0)
 *
 * PROBLEM==201 additionally emits (after the steps):
 *   ct.kgld     (6/3)  facedim,ix,iy,fB1,fB2,fB3 -> fB1',fB2',fB3'
 *               (x-faces ix∈[0,NX]×iy∈[-1,NY]; y-faces ix∈[-1,NX]×iy∈[0,NY],
 *                PRNG-filled B rows, then flux_ct(); unwritten faces pass
 *                through unchanged)
 *   bfroma.kgld (5/3)  ix,iy,A1,A2,A3 -> B1,B2,B3
 *               (PRNG A at domain cell centers, set_bc, calc_BfromA(p,1))
 */

#include "ko.h"
#include <stdint.h>
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

static uint64_t rngstate;
static double rnd(void)
{
  rngstate ^= rngstate >> 12;
  rngstate ^= rngstate << 25;
  rngstate ^= rngstate >> 27;
  return (double)((rngstate * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
}
static double rnd_pm(void) { return 2. * rnd() - 1.; }

static void dump_step(FILE *f, ldouble t, ldouble dtused)
{
  int ix, iy, iz, iv;
  double v;
  v = t; fwrite(&v, 8, 1, f);
  v = dtused; fwrite(&v, 8, 1, f);
  for (iz = 0; iz < NZ; iz++)
    for (iy = 0; iy < NY; iy++)
      for (ix = 0; ix < NX; ix++)
        for (iv = 0; iv < NV; iv++)
        { v = get_u(u, iv, ix, iy, iz); fwrite(&v, 8, 1, f); }
  for (iz = 0; iz < NZ; iz++)
    for (iy = 0; iy < NY; iy++)
      for (ix = 0; ix < NX; ix++)
        for (iv = 0; iv < NV; iv++)
        { v = get_u(p, iv, ix, iy, iz); fwrite(&v, 8, 1, f); }
  for (iz = 0; iz < NZ; iz++)
    for (iy = 0; iy < NY; iy++)
      for (ix = 0; ix < NX; ix++)
      {
        v = (double)get_cflag(ENTROPYFLAG, ix, iy, iz); fwrite(&v, 8, 1, f);
        v = (double)get_cflag(HDFIXUPFLAG, ix, iy, iz); fwrite(&v, 8, 1, f);
#ifdef RADIATION
        v = (double)get_cflag(RADFIXUPFLAG, ix, iy, iz); fwrite(&v, 8, 1, f);
        v = (double)get_cflag(RADIMPFIXUPFLAG, ix, iy, iz); fwrite(&v, 8, 1, f);
#endif
      }
}

int main(int argc, char **argv)
{
  const char *outdir = argc > 1 ? argv[1] : ".";
  const char *name = argc > 2 ? argv[2] : "step.kstp";
  int nsteps = argc > 3 ? atoi(argv[3]) : 10;
  int ii, ix, iy, iz, iv;

#ifdef RADIATION
  /* Zig's LU mirrors GSL-with-handler-off (silent inf on exactly singular
   * FD Jacobians, rejected by the damping ladder); production C would
   * abort here. Same call as oracle/harness_implicit.c. */
  gsl_set_error_handler_off();
#endif

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  set_grid(&min_dx, &min_dy, &min_dz, &max_dt); /* globals, as ko.c:84 */
  alloc_loops();
  calc_metric();

  /* ---- ko.c init sequence ---- */
  ldouble t = 0., t1 = TMAX;
  set_initial_profile();
  set_bc(t, 1);
#ifdef MAGNFIELD
#ifdef VECPOTGIVEN
  calc_BfromA(p, 1);
  set_bc(t, 1);
#endif
#endif

  /* ---- solve_the_problem preamble (problem.c:59-82) ---- */
  set_gammagas(0);
  dt = -1.;
  max_ws[0] = max_ws[1] = max_ws[2] = 10000.;
  if (NZ > 1)
    tstepdenmax = max_ws[0] / min_dx + max_ws[1] / min_dy + max_ws[2] / min_dz;
  else if (NY > 1)
    tstepdenmax = max_ws[0] / min_dx + max_ws[1] / min_dy;
  else
    tstepdenmax = max_ws[0] / min_dx;
  tstepdenmax /= TSTEPLIM;
  tstepdenmin = tstepdenmax;

  for (ii = 0; ii < Nloop_0; ii++)
  {
    ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
    set_u_scalar(cell_tstepden, ix, iy, iz, tstepdenmax);
    set_u_scalar(cell_dt, ix, iy, iz, 1. / tstepdenmax);
  }

  /* ---- KSTP ---- */
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", outdir, name);
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  {
    uint32_t version = 1, d32;
    uint64_t nrec = (uint64_t)nsteps + 1;
    fwrite("KSTP", 1, 4, f);
    fwrite(&version, 4, 1, f);
    d32 = NX; fwrite(&d32, 4, 1, f);
    d32 = NY; fwrite(&d32, 4, 1, f);
    d32 = NZ; fwrite(&d32, 4, 1, f);
    d32 = NV; fwrite(&d32, 4, 1, f);
    fwrite(&nrec, 8, 1, f);
  }
  dump_step(f, t, 0.);

  /* ---- main loop: TIMESTEPPING==RK2IMEX block, problem.c:141-402 ---- */
  nstep = 0;
  int istep;
  for (istep = 0; istep < nsteps; istep++)
  {
      global_int_slot[GLOBALINTSLOT_NTOTALRADIMPFIXUPS] = 0;
      global_int_slot[GLOBALINTSLOT_NTOTALMHDFIXUPS] = 0;
      global_int_slot[GLOBALINTSLOT_NTOTALRADFIXUPS] = 0;

      global_time = t;
      nstep++;

      mpi_synchtiming(&t);

      dt = 1. / tstepdenmax;
      global_dt = dt;
      if (t + dt > t1) dt = t1 - t;

      tstepdenmax = -1.;
      tstepdenmin = BIG;

#if (RADVISCOSITY == SHEARVISCOSITY)
      calc_Rij_visc_total();
#endif

      /* RK2IMEX */
      {
	    my_finger(global_time);

	    ldouble gamma = 1. - 1. / sqrt(2.);
	    ldouble dtcell;

	    save_timesteps();
	    set_gammagas(0);

	    dtcell = dt;

	    /******* 1st implicit **********/
	    copyi_u(1., u, ut0);
	    copy_u(1., p, ptm1);
	    op_implicit(t, dt * gamma);
	    global_impdt = dt * gamma;
	    copyi_u(1., p, ppostimplicit);

	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(drt1, iv, ix, iy, iz, (1. / (dtcell * gamma)) * get_u(u, iv, ix, iy, iz) + (-1. / (dtcell * gamma)) * get_u(ut0, iv, ix, iy, iz));
	    }

	    /******* 1st explicit **********/
	    copyi_u(1., u, ut1);
	    calc_u2p(0, 1);
	    do_correct();
	    mpi_exchangedata();
	    set_bc(t, 0);
	    op_explicit(t, dt);
	    apply_dynamo(t, dt);
	    op_intermediate(t, dt);
	    global_expdt = dt;
	    copy_entropycount();

	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(dut1, iv, ix, iy, iz, (1. / (dtcell)) * get_u(u, iv, ix, iy, iz) + (-1. / (dtcell)) * get_u(ut1, iv, ix, iy, iz));
	    }

	    /******* 1st together **********/
	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(u, iv, ix, iy, iz, get_u(ut0, iv, ix, iy, iz) + (dtcell) * get_u(dut1, iv, ix, iy, iz) + (dtcell * (1. - 2. * gamma)) * get_u(drt1, iv, ix, iy, iz));
	    }

	    /******* 2nd implicit **********/
	    copyi_u(1., u, uforget);
	    calc_u2p(0, 1);
	    copy_u(1., p, ptm1);
	    do_correct();
	    op_implicit(t, gamma * dt);
	    global_impdt = gamma * dt;
	    copyi_u(1., p, ppostimplicit);

	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(drt2, iv, ix, iy, iz, (1. / (dtcell * gamma)) * get_u(u, iv, ix, iy, iz) + (-1. / (dtcell * gamma)) * get_u(uforget, iv, ix, iy, iz));
	    }

	    /******* 2nd explicit **********/
	    copyi_u(1., u, ut2);
	    calc_u2p(0, 1);
	    do_correct();
	    mpi_exchangedata();
	    set_bc(t, 0);
	    op_explicit(t, dt);
	    apply_dynamo(t, dt);
	    op_intermediate(t, dt);
	    global_expdt = dt;

	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(dut2, iv, ix, iy, iz, (1. / (dtcell)) * get_u(u, iv, ix, iy, iz) + (-1. / (dtcell)) * get_u(ut2, iv, ix, iy, iz));
	    }

	    /******* explicit together **********/
	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(u, iv, ix, iy, iz, get_u(ut0, iv, ix, iy, iz) + (dtcell / 2.) * get_u(dut1, iv, ix, iy, iz) + (dtcell / 2.) * get_u(dut2, iv, ix, iy, iz));
	    }

	    /******* implicit together ***********/
	    for (ii = 0; ii < Nloop_0; ii++)
	    {
	      ix = loop_0[ii][0]; iy = loop_0[ii][1]; iz = loop_0[ii][2];
	      PLOOP(iv) set_u(u, iv, ix, iy, iz, get_u(u, iv, ix, iy, iz) + (dtcell / 2.) * get_u(drt1, iv, ix, iy, iz) + (dtcell / 2.) * get_u(drt2, iv, ix, iy, iz));
	    }

	    calc_u2p(0, 1);
	    do_correct();
	    mpi_exchangedata();
	    set_bc(t, 0);

	    t += dt;
	    update_entropy();
      }

      dump_step(f, t, dt);
  }
  fclose(f);
  printf("harness_step: %s written (%d steps)\n", name, nsteps);

#if (PROBLEM == 201)
  /* ---- flux_ct golden ------------------------------------------------ */
  {
    /* faces the EMFs read: x-faces ix∈[0,NX]×iy∈[-1,NY]; y ix∈[-1,NX]×iy∈[0,NY] */
    uint64_t nrec = (uint64_t)(NX + 1) * (NY + 2) + (uint64_t)(NX + 2) * (NY + 1);
    FILE *fc = open_kgld(outdir, "ct.kgld", nrec, 6, 3);

    rngstate = 0x5a49474354303601ULL;
    for (iy = -1; iy <= NY; iy++)
      for (ix = 0; ix <= NX; ix++)
      {
        set_ubx(flbx, B1, ix, iy, 0, rnd_pm());
        set_ubx(flbx, B2, ix, iy, 0, rnd_pm());
        set_ubx(flbx, B3, ix, iy, 0, rnd_pm());
      }
    for (iy = 0; iy <= NY; iy++)
      for (ix = -1; ix <= NX; ix++)
      {
        set_uby(flby, B1, ix, iy, 0, rnd_pm());
        set_uby(flby, B2, ix, iy, 0, rnd_pm());
        set_uby(flby, B3, ix, iy, 0, rnd_pm());
      }

    flux_ct();

    /* replay the PRNG for the inputs */
    rngstate = 0x5a49474354303601ULL;
    for (iy = -1; iy <= NY; iy++)
      for (ix = 0; ix <= NX; ix++)
      {
        double in[6], out[3];
        in[0] = 0.; in[1] = (double)ix; in[2] = (double)iy;
        in[3] = rnd_pm(); in[4] = rnd_pm(); in[5] = rnd_pm();
        out[0] = get_ub(flbx, B1, ix, iy, 0, 0);
        out[1] = get_ub(flbx, B2, ix, iy, 0, 0);
        out[2] = get_ub(flbx, B3, ix, iy, 0, 0);
        fwrite(in, 8, 6, fc);
        fwrite(out, 8, 3, fc);
      }
    for (iy = 0; iy <= NY; iy++)
      for (ix = -1; ix <= NX; ix++)
      {
        double in[6], out[3];
        in[0] = 1.; in[1] = (double)ix; in[2] = (double)iy;
        in[3] = rnd_pm(); in[4] = rnd_pm(); in[5] = rnd_pm();
        out[0] = get_ub(flby, B1, ix, iy, 0, 1);
        out[1] = get_ub(flby, B2, ix, iy, 0, 1);
        out[2] = get_ub(flby, B3, ix, iy, 0, 1);
        fwrite(in, 8, 6, fc);
        fwrite(out, 8, 3, fc);
      }
    fclose(fc);
  }

  /* ---- calc_BfromA golden -------------------------------------------- */
  {
    uint64_t nrec = (uint64_t)NX * NY;
    FILE *fb = open_kgld(outdir, "bfroma.kgld", nrec, 5, 3);

    rngstate = 0x5a4947424652410aULL;
    for (iz = 0; iz < NZ; iz++)
      for (iy = 0; iy < NY; iy++)
        for (ix = 0; ix < NX; ix++)
        {
          set_u(p, B1, ix, iy, iz, 0.1 * rnd_pm());
          set_u(p, B2, ix, iy, iz, 0.1 * rnd_pm());
          set_u(p, B3, ix, iy, iz, 0.1 * rnd_pm());
        }
    set_bc(t, 1);
    calc_BfromA(p, 1);

    rngstate = 0x5a4947424652410aULL;
    for (iz = 0; iz < NZ; iz++)
      for (iy = 0; iy < NY; iy++)
        for (ix = 0; ix < NX; ix++)
        {
          double in[5], out[3];
          in[0] = (double)ix; in[1] = (double)iy;
          in[2] = 0.1 * rnd_pm(); in[3] = 0.1 * rnd_pm(); in[4] = 0.1 * rnd_pm();
          out[0] = get_u(p, B1, ix, iy, iz);
          out[1] = get_u(p, B2, ix, iy, iz);
          out[2] = get_u(p, B3, ix, iy, iz);
          fwrite(in, 8, 5, fb);
          fwrite(out, 8, 3, fb);
        }
    fclose(fb);
    printf("harness_step: ct.kgld + bfroma.kgld written\n");
  }
#endif /* PROBLEM == 201 */

  return 0;
}
