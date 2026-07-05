// ZIGRADTUBE (PROBLEM 203) — Farris et al. (2008) radiative shock tube 2
// (mildly relativistic strong shock, Sądowski et al. 2013 Table 3) for
// koral-zig M10 step tests. 64 cells over x ∈ [-20, 20], MINK, PPM +
// RK2IMEX + LAXF, fixed (Dirichlet) x-boundaries via bc.c.
//
// Radiation: M1 + the PUFFY implicit block (RADIMP_START_WITH_BISECT,
// SCALE_JACOBIAN, ALLOWRADCEILINGINIMPLICIT, ALLOWFORENTRINF4DPRIM) with
// grey absorption kappa = 0.2·rho via PR_KAPPA (kappa.c) and no scattering.
// Rad floors/ceilings left at choices.h defaults (GAMMAMAXRAD 100 covers
// the tube's γ ≈ 1.03).
//
// MASS makes KORAL's physical-constant LTE the paper's a = Ê_L/(p_L/ρ_L)⁴:
// a_code = 4σ_gu·(μ m_p c²/k_B)⁴ ∝ MASS² (see koral/testing/tubes.zig,
// test "tube MASS values"). Tube 2: a_paper = 7.8125e4.

#define MU_GAS 1.
#define MU_I 1.
#define MU_E 1.

#define NORESTART

/************************************/
//radiation
/************************************/
#define RADIATION

#define U2PCONV 1.e-10
#define RADIMPLICITDAMPINGFACTOR 3.
#define RADIMPLICITMAXENCHANGEDOWN 10.
#define RADIMPLICIMAXENCHANGEUP 10.
#define MAXRADIMPDAMPING 1.e-3
#define RADIMPCONV 1.e-10
#define RADIMPEPS 1.e-6
#define RADIMPMAXITER 40
#define RADIMPCONVREL 1.e-8
#define RADIMPCONVRELERR 1.e-4
#define RADIMPCONVRELENTR 1.e-4
#define RADIMPCONVRELENTRERR 1.e-2
#define OPDAMPINIMPLICIT 0

#define RADIMP_START_WITH_BISECT
#define ALLOWRADCEILINGINIMPLICIT
#define ALLOWFORENTRINF4DPRIM
#define SCALE_JACOBIAN

/************************************/
//reconstruction / Courant
/************************************/
#define INT_ORDER 2
#define TIMESTEPPING RK2IMEX
#define TSTEPLIM .5
#define FLUXLIMITER 0
#define MINMOD_THETA 1.5
#define DOFIXUPS 1
#define DOU2PRADFIXUPS 0
#define DOU2PMHDFIXUPS 1
#define DORADIMPFIXUPS 0

#define B2RHOFLOORFRAME DRIFTFRAME

/************************************/
//blackhole (units only; tuned so code LTE == paper LTE, tubes.zig)
/************************************/
#define MASS 6.38448437172474900e2
#define BHSPIN 0.

/************************************/
//coordinates / resolution
/************************************/
#define MYCOORDS MINKCOORDS
#define MINX -20.
#define MAXX 20.
#define MINY 0.
#define MAXY 1.
#define MINZ 0.
#define MAXZ 1.

#define TNX 64
#define TNY 1
#define TNZ 1
#define NTX 1
#define NTY 1
#define NTZ 1

// bc.c pins the asymptotic states (Dirichlet); SPECIFIC_BC routes set_bc
// through calc_bc → PR_BC (only x has ghosts: TNY = TNZ = 1)
#define SPECIFIC_BC

/************************************/
//physics
/************************************/
#define GAMMA (5./3.)

/************************************/
//tube 2 states (Sądowski et al. 2013 Table 3)
/************************************/
#define RADTUBE_KAPPA 0.2
#define RADTUBE_RHO_L 1.0
#define RADTUBE_P_L 4.0e-3
#define RADTUBE_UX_L 0.25
#define RADTUBE_E_L 2.0e-5
#define RADTUBE_RHO_R 3.11
#define RADTUBE_P_R 4.512e-2
#define RADTUBE_UX_R 8.04e-2
#define RADTUBE_E_R 3.46e-3

/************************************/
//output (all off; the harness dumps KSTP itself)
/************************************/
#define DTOUT1 1.e50
#define DTOUT2 1.e50
#define TMAX 1.e50
#define OUTCOORDS MINKCOORDS
#define OUTVEL VEL4
#define ALLSTEPSOUTPUT 0
#define NOUTSTOP 100000
#define SILOOUTPUT 0
#define OUTOUTPUT 0
#define COORDOUTPUT 0
#define SIMOUTPUT 0
#define RADOUTPUT 0
#define SCAOUTPUT 0
#define AVGOUTPUT 0
#define THOUTPUT 0
