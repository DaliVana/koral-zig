// ZIGRADPULSE bc — unused (COPY_XBC handles the boundaries); calc_bc()
// includes this unconditionally, so keep it compilable: plain outflow copy.

int iix = ix, iiy = iy, iiz = iz;
if (iix < 0) iix = 0;
if (iix >= NX) iix = NX - 1;

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = get_u(p, iv, iix, iiy, iiz);

struct geometry geom;
fill_geometry(ix, iy, iz, &geom);
p2u(pp, uu, &geom);
