// ZIGPOLE bc — x: outflow (nearest domain cell), y: transmitting pole,
// z: periodic. The y branch is PROBLEMS/KATOTORUS_TILTED/bc.c:226-250 with
// the theta components flipped unconditionally, as mpi.c:1984-2062 does
// between tiles (the serial hook leaves them unflipped while ifinit).

int iix = ix, iiy = iy, iiz = iz, iv;
struct geometry geom;
fill_geometry(ix, iy, iz, &geom);

if (BCtype == XBCLO || BCtype == XBCHI)
{
  iix = (BCtype == XBCLO) ? 0 : NX - 1;
  for (iv = 0; iv < NV; iv++) pp[iv] = get_u(p, iv, iix, iiy, iiz);
  p2u(pp, uu, &geom);
  return 0;
}

if (BCtype == YBCLO || BCtype == YBCHI)
{
  iiy = (BCtype == YBCLO) ? -iy - 1 : NY - (iy - NY) - 1;
  iiz = iz + (TNZ / 2);
  if (iiz > TNZ - 1) iiz -= TNZ;
  for (iv = 0; iv < NV; iv++)
  {
    if (iv == VY || iv == B2 || iv == VZ || iv == B3)
      pp[iv] = -get_u(p, iv, iix, iiy, iiz);
    else
      pp[iv] = get_u(p, iv, iix, iiy, iiz);
  }
  p2u(pp, uu, &geom);
  return 0;
}

// periodic in phi
if (iiz < 0) iiz += NZ;
if (iiz > NZ - 1) iiz -= NZ;
for (iv = 0; iv < NV; iv++) pp[iv] = get_u(p, iv, iix, iiy, iiz);
p2u(pp, uu, &geom);
return 0;
