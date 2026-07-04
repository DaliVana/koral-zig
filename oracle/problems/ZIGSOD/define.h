// ZIGSOD (PROBLEM 200) — 1D SR Sod tube for koral-zig M5 step tests.
// 64 cells, MINK, PPM + RK2IMEX + LAXF, outflow (COPY) x-boundaries.
// Floors left at choices.h defaults except the drift-frame B-floor frame
// (matches the Zig implementation; irrelevant without MAGNFIELD).

#define MU_GAS 1.
#define MU_I 1.
#define MU_E 1.

#define NORESTART

#define U2PCONV 1.e-10

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
#define GAMMA (5./3.)

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
