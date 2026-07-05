// ZIGRADTUBE init — Farris tube 2: left/right LTE states split at x = 0,
// radiation comoving with the gas (F̂ = 0 ⟺ rad velocity = gas velocity).

ldouble uu[NV], pp[NV];
struct geometry geom;
fill_geometry(ix, iy, iz, &geom);

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

if (geom.xx < 0.) {
  pp[RHO] = RADTUBE_RHO_L;
  pp[UU] = RADTUBE_P_L / (GAMMA - 1.);
  pp[VX] = RADTUBE_UX_L;
  pp[EE0] = RADTUBE_E_L;
  pp[FX0] = RADTUBE_UX_L;
} else {
  pp[RHO] = RADTUBE_RHO_R;
  pp[UU] = RADTUBE_P_R / (GAMMA - 1.);
  pp[VX] = RADTUBE_UX_R;
  pp[EE0] = RADTUBE_E_R;
  pp[FX0] = RADTUBE_UX_R;
}
pp[VY] = pp[VZ] = 0.;
pp[FY0] = pp[FZ0] = 0.;
pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], ix, iy, iz);

p2u(pp, uu, &geom);

for (iv = 0; iv < NV; iv++) {
  set_u(u, iv, ix, iy, iz, uu[iv]);
  set_u(p, iv, ix, iy, iz, pp[iv]);
}
