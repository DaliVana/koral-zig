// ZIGRADPULSE kappa (PR_KAPPA) — grey absorption; slots assigned explicitly
// (see ZIGRADTUBE/kappa.c for why).

kappa = RADPULSE_KAPPA * rho;
opac->kappaGasAbs = opac->kappaRadAbs = kappa;
opac->kappaGasNum = opac->kappaRadNum = kappa;
opac->kappaGasRoss = opac->kappaRadRoss = kappa;
