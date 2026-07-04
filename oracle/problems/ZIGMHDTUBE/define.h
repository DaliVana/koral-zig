// ZIGMHDTUBE (PROBLEM 202) — 1D SR Brio–Wu / Balsara-1 for koral-zig M6
// step tests. 64 cells, MINK, Γ=2, direct B init (no vector potential),
// PPM + RK2IMEX + LAXF, outflow (COPY) x-boundaries.

#define MU_GAS 1.
#define MU_I 1.
#define MU_E 1.

#define NORESTART

#define U2PCONV 1.e-10

/************************************/
//magnetic choices
/************************************/
#define MAGNFIELD
#define GDETIN 1

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
//blackhole (units only)
/************************************/
#define MASS 10.
#define BHSPIN 0.

/************************************/
//coordinates / resolution
/************************************/
#define MYCOORDS MINKCOORDS
#define MINX 0.
#define MAXX 1.
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
#define GAMMA (2.)

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
