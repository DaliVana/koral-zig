// ZIGRADPULSE init — LTE temperature bump at rest: T = p/ρ (paper units),
// E = a·T⁴ exactly matches KORAL's 4σT⁴ through the MASS choice.

ldouble uu[NV], pp[NV];
struct geometry geom;
fill_geometry(ix, iy, iz, &geom);

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

ldouble tpap = RADPULSE_T0 * (1. + RADPULSE_AMP * exp(-geom.xx * geom.xx / (RADPULSE_W * RADPULSE_W)));

pp[RHO] = RADPULSE_RHO;
pp[UU] = RADPULSE_RHO * tpap / (GAMMA - 1.);
pp[VX] = pp[VY] = pp[VZ] = 0.;
pp[EE0] = RADPULSE_APAPER * tpap * tpap * tpap * tpap;
pp[FX0] = pp[FY0] = pp[FZ0] = 0.;
pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], ix, iy, iz);

p2u(pp, uu, &geom);

for (iv = 0; iv < NV; iv++) {
  set_u(u, iv, ix, iy, iz, uu[iv]);
  set_u(p, iv, ix, iy, iz, pp[iv]);
}
