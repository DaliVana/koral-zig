// ZIGOT bc — unused (PERIODIC_XBC/PERIODIC_YBC); keep calc_bc compilable.

int iix = ix, iiy = iy, iiz = iz;
if (iix < 0) iix += NX;
if (iix >= NX) iix -= NX;
if (iiy < 0) iiy += NY;
if (iiy >= NY) iiy -= NY;

int iv;
for (iv = 0; iv < NV; iv++) pp[iv] = get_u(p, iv, iix, iiy, iiz);

struct geometry geom;
fill_geometry(ix, iy, iz, &geom);
p2u(pp, uu, &geom);
