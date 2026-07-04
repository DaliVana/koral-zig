// ZIGMHDTUBE init — Balsara test 1 (SR Brio–Wu), Γ=2:
//   left:  rho=1,     p=1,   By=+1
//   right: rho=0.125, p=0.1, By=−1
//   Bx=0.5 everywhere, at rest.

ldouble uu[NV], pp[NV];
struct geometry geom;
fill_geometry(ix, iy, iz, &geom);

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

if (geom.xx < 0.5) {
  pp[RHO] = 1.0;
  pp[UU] = 1.0 / (GAMMA - 1.);
  pp[B2] = 1.0;
} else {
  pp[RHO] = 0.125;
  pp[UU] = 0.1 / (GAMMA - 1.);
  pp[B2] = -1.0;
}
pp[VX] = pp[VY] = pp[VZ] = 0.;
pp[B1] = 0.5;
pp[B3] = 0.;
pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], ix, iy, iz);

p2u(pp, uu, &geom);

for (iv = 0; iv < NV; iv++) {
  set_u(u, iv, ix, iy, iz, uu[iv]);
  set_u(p, iv, ix, iy, iz, pp[iv]);
}
