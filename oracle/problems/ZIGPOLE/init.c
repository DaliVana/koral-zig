// ZIGPOLE init — a smooth, non-axisymmetric magnetised shell. Nothing here
// is an equilibrium: the test is a few forced steps of identical arithmetic
// in C and Zig, on a state that exercises every sign and index of the
// transmitting pole (phi-dependent scalars, v^theta and v^phi through the
// axis, all three field components from a vector potential in the B slots).

ldouble uu[NV], pp[NV];
struct geometry geom, geomBL;
fill_geometry(ix, iy, iz, &geom);
fill_geometry_arb(ix, iy, iz, &geomBL, BLCOORDS);

ldouble r = geomBL.xx;
ldouble th = geomBL.yy;
ldouble ph = geomBL.zz;
ldouble st = sin(th), ct = cos(th);

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = 0.;

pp[RHO] = (1. + 0.3 * ct + 0.2 * st * cos(ph - 0.7)) * pow(r / 4., -1.5);
pp[UU] = 0.02 * pp[RHO] * (1. + 0.1 * st * sin(ph + 0.2)) * (4. / r);
// VELPRIM == VELR, components in MKS2: x1 = log(r - r0), so u^x1 = u^r /
// (r - r0); theta is stretched by up to ~30 at the poles (MKSH0 = 0.9), hence
// the small u^x2. Everything is subsonic to mildly supersonic.
pp[VX] = -0.2 * (1. + 0.2 * st * cos(ph)) / (r - MKSR0);
pp[VY] = (0.004 * st * sin(ph + 0.5) + 0.002 * st * ct) / r;
pp[VZ] = (0.02 + 0.01 * cos(ph - 1.1)) / r;
pp[ENTR] = calc_Sfromu(pp[RHO], pp[UU], ix, iy, iz);

// vector potential (VECPOTGIVEN): sin^2(theta) keeps B finite on the axis.
// calc_BfromA drops the d/dphi terms of the curl, so only A_phi may depend
// on phi if the field is to come out solenoidal: B^r and B^theta then vary
// in phi, B^phi (from A_r) does not.
pp[B1] = 0.004 * st * st * (1. + 0.3 * ct);
pp[B2] = 0.;
pp[B3] = 0.03 * r * st * st * (1. + 0.4 * st * cos(ph + 0.3));

p2u(pp, uu, &geom);

for (iv = 0; iv < NV; iv++) {
  set_u(u, iv, ix, iy, iz, uu[iv]);
  set_u(p, iv, ix, iy, iz, pp[iv]);
}
