// ZIGRADPULSE (PROBLEM 204) — optically thick LTE radiation pulse for
// koral-zig M10 step tests: a Gaussian TEMPERATURE bump with gas and
// radiation in LTE everywhere (a cold pulse on a non-LTE background
// collapses: wing cells go M1-inconsistent and pin the γ_rad ceiling),
// diffusing under grey ABSORPTION kappa = 10·rho via PR_KAPPA.
// 64 cells over x ∈ [-50, 50], MINK, PPM + RK2IMEX + LAXF, outflow x-BCs.
// τ_cell ≈ 16, τ_domain = 1000; exercises the τ-limited M1 wavespeeds and
// a stiffer implicit (χΔt ≈ 4 at CFL) than the radtube.
//
// Paper units: T = p/ρ = 0.01·(1 + 0.5·exp(−x²/w²)), E = a·T⁴ with
// a = 5e5 realized through MASS (koral/testing/tubes.zig `pulse`).

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
#define MASS 1.61516098403194600e3
#define BHSPIN 0.

/************************************/
//coordinates / resolution
/************************************/
#define MYCOORDS MINKCOORDS
#define MINX -50.
#define MAXX 50.
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

#define COPY_XBC

/************************************/
//physics
/************************************/
#define GAMMA (5./3.)

/************************************/
//pulse parameters (paper units, cf. koral/testing/tubes.zig `pulse`)
/************************************/
#define RADPULSE_KAPPA 10.0
#define RADPULSE_RHO 1.0
#define RADPULSE_T0 0.01
#define RADPULSE_AMP 0.5
#define RADPULSE_W 10.0
#define RADPULSE_APAPER 5.0e5

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
