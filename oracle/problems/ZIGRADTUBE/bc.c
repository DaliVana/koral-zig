// ZIGRADTUBE bc — Dirichlet: ghosts pinned to the asymptotic tube states
// (the gas streams in from the left and out on the right; outflow-copy
// would let the inflow state drift).

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

if (ix < 0) {
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

struct geometry geom;
fill_geometry(ix, iy, iz, &geom);
p2u(pp, uu, &geom);
