// ZIGRADTUBE kappa (PR_KAPPA) — grey absorption, kappa = RADTUBE_KAPPA·rho
// in geometrical units. The opac slots are assigned EXPLICITLY: the
// calc_kappa_from_state wrapper's `opac->kappaGasAbs >= 0.` broadcast test
// otherwise reads uninitialized stack (cf. LRTORUS/kappa.c).

kappa = RADTUBE_KAPPA * rho;
opac->kappaGasAbs = opac->kappaRadAbs = kappa;
opac->kappaGasNum = opac->kappaRadNum = kappa;
opac->kappaGasRoss = opac->kappaRadRoss = kappa;
