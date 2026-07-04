// ZIGOT init — SR Orszag–Tang (Beckwith & Stone 2011 flavor), v0 = 0.5.
// B slots carry the vector potential A_i (VECPOTGIVEN):
//   A3 = −B0(cos(2x)/2 + cos y)  →  B = B0(−sin y, sin 2x, 0)

ldouble uu[NV], pp[NV];
struct geometry geom;
fill_geometry(ix, iy, iz, &geom);

ldouble xx = geom.xx;
ldouble yy = geom.yy;

ldouble rho = 25. / (36. * M_PI);
ldouble pgas = 5. / (12. * M_PI);
ldouble b0 = 1. / sqrt(4. * M_PI);
ldouble vx = -0.5 * sin(yy);
ldouble vy = 0.5 * sin(xx);
ldouble gl = 1. / sqrt(1. - vx * vx - vy * vy);

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

pp[RHO] = rho;
pp[UU] = pgas / (GAMMA - 1.);
pp[VX] = gl * vx; // VELPRIM == VELR
pp[VY] = gl * vy;
pp[VZ] = 0.;
pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], ix, iy, iz);

pp[B1] = 0.;
pp[B2] = 0.;
pp[B3] = -b0 * (0.5 * cos(2. * xx) + cos(yy));

p2u(pp, uu, &geom);

for (iv = 0; iv < NV; iv++) {
  set_u(u, iv, ix, iy, iz, uu[iv]);
  set_u(p, iv, ix, iy, iz, pp[iv]);
}
