// ZIGPOLE (PROBLEM 205) — the transmitting polar axis (TRANSMITTING_YBC) for
// the koral-zig step test. A small 3D MKS2 shell around a spinning hole,
// full 2 pi in phi, hydro + MHD (no radiation, so the comparison is not
// buried under the implicit solver's chaos), a smooth state that is NOT
// axisymmetric and carries velocity and field through both poles.
// PPM + RK2IMEX + LAXF; x: outflow copy, y: transmitting, z: periodic.
// TRANSMITTING_YBC alone switches on CORRECT_POLARAXIS, CORRECTMAGNFIELD
// and NCCORRECTPOLAR 1 (choices.h:273-292).

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
#define VECPOTGIVEN

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
//the pole
/************************************/
#define TRANSMITTING_YBC

/************************************/
//blackhole
/************************************/
#define MASS 10.
#define BHSPIN 0.5

/************************************/
//coordinates / resolution (PUFFY's MKS2 map)
/************************************/
#define MKSR0 0.1
#define MKSH0 0.9
#define METRICAXISYMMETRIC
#define RMIN 4.
#define RMAX 40.
#define MYCOORDS MKS2COORDS
#define MINX (log(RMIN-MKSR0))
#define MAXX (log(RMAX-MKSR0))
#define MINY (0.005)
#define MAXY (1.-0.005)
#define MINZ (-M_PI)
#define MAXZ (M_PI)

#define TNX 12
#define TNY 12
#define TNZ 8
#define NTX 1
#define NTY 1
#define NTZ 1

#define SPECIFIC_BC
#define PERIODIC_ZBC

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
#define OUTCOORDS BLCOORDS
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
