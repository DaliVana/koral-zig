/* M13 oracle: the full PUFFY pipeline, forced-dt step test (KSTP).
 *
 * Built against a reduced-grid PUFFY (gen_golden.sh seds TNX/TNY down in the
 * zigpuffy variant so the golden is committable). Runs the complete ko.c init
 * sequence — PR_PREPINIT (limotorus) + set_initial_profile + set_bc +
 * calc_BfromA + set_bc + PR_POSTINIT (beta normalization) — i.e. the true
 * post-init state a production run starts from, then the problem.c:141-402
 * RK2IMEX block verbatim (radiation implicit source, RADVISCOSITY, dynamo),
 * dumping the full domain each step.
 *
 * KSTP format (identical to harness_step.c): "KSTP" u32 v=1, u32 nx,ny,nz,nv,
 * u64 nrec(=nsteps+1); per record f64 t, f64 dt, u[...], p[...], flags[...]
 * (iv fastest; RADIATION → nf=4: ENTROPY, HDFIXUP, RADFIXUP, RADIMPFIXUP).
 */

#include "ko.h"
#include <stdint.h>
#include <gsl/gsl_errno.h>

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
  const char *name = argc > 2 ? argv[2] : "puffy.kstp";
  int nsteps = argc > 3 ? atoi(argv[3]) : 4;
  int ii, ix, iy, iz, iv;

  /* silent-inf LU on singular FD Jacobians, matching Zig / the other harnesses */
  gsl_set_error_handler_off();

  global_time = 0.;
  init_pointers();
  initialize_arrays();
  initialize_constants();
  set_grid(&min_dx, &min_dy, &min_dz, &max_dt);
  alloc_loops();
  calc_metric();

  /* ---- full ko.c init sequence (post-init state) ---- */
  ldouble t = 0., t1 = TMAX;
  {
#include PR_PREPINIT
  }
  set_initial_profile();
  set_bc(t, 1);
#ifdef MAGNFIELD
#ifdef VECPOTGIVEN
  calc_BfromA(p, 1);
  set_bc(t, 1);
#endif
#endif
  {
#include PR_POSTINIT
  }

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

  /* ---- KSTP header + record 0 ---- */
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

  /* ---- RK2IMEX loop (problem.c:141-402, serial path) ---- */
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

      {
        my_finger(global_time);
        ldouble gamma = 1. - 1. / sqrt(2.);
        ldouble dtcell = dt;

        save_timesteps();
        set_gammagas(0);

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
      printf("harness_puffy_step: step %d/%d  t=%.6e dt=%.6e\n", istep + 1, nsteps, t, dt);
  }
  fclose(f);
  printf("harness_puffy_step: %s written (%d steps, %dx%d)\n", name, nsteps, NX, NY);
  return 0;
}
