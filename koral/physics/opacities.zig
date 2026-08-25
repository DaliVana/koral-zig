//! Opacities (C: opacities.c); the channels the PUFFY build compiles:
//! BREMSSTRAHLUNG + SYNCHROTRON absorption (calc_opacities_from_state) and
//! the PUFFY scattering hook (PROBLEMS/PUFFY/kappaes.c) with its
//! KLEINNISHINA correction. Everything here is scalar-in/scalar-out; the
//! pp/geometry-level wrappers live in physics/radforce.zig.
//!
//! Physics: with gas (T_e) and radiation (T_rad) temperatures decoupled, a
//! single grey opacity is not enough; the code carries a family of means:
//! gas_abs (Planck at T_e) sets emission j = κ·4πB(T_e); rad_abs (Planck
//! weighted by the radiation spectrum) sets absorption, the ζ = T_rad/T_e
//! factors interpolating between the two; gas/rad_ross (Rosseland) govern
//! diffusion, the wavespeed τ-limiter and the radviscosity mfp; rad_num
//! (photon-number-weighted) matters only when photon number is evolved.
//! Bremsstrahlung is the classic ε_ff ∝ n_e n_i √T_e with Gaunt factor 1.2
//! and a mild relativistic correction, turned into an opacity via
//! Kirchhoff's law (divide by the blackbody energy density). Thermal
//! synchrotron absorption is a fitted function of zbr = kT_rad/(hν_crit),
//! ν_crit ∝ T_e²B; typical photon energy over the synchrotron critical
//! frequency: suppressed for nonrelativistic electrons by Θ²/(1+Θ²),
//! Θ = kT_e/m_ec² (thermal synchrotron dies for cold electrons). Electron
//! scattering is Thomson κ_es = 0.2(1+X) cm²/g with a Klein-Nishina
//! reduction once T_rad approaches m_ec²/k. τ = χ·dx are the per-cell
//! optical depths.
//!
//! C-fidelity notes:
//!  * PUFFY does NOT wire PR_KAPPA; PROBLEMS/PUFFY/kappa.c is dead code;
//!    absorption comes from the default calc_opacities_from_state.
//!  * USE_SYNCHROTRON_BRIDGE_FUNCTIONS is commented out of choices.h
//!    entirely (PUFFY's NO_SYNCHROTRON_BRIDGE_FUNCTIONS is vestigial), so
//!    the bridge blocks are off and the Terelfactor suppression is active.
//!  * The C body computes Trad_lim/nph_lim from pp[NF] even without
//!    EVOLVEPHOTONNUMBER: unused locals reading a stale slot; skipped.
//!  * calc_opacities_from_state uses the misc.c *globals* (kappacgs2gu,
//!    rhogu2cgs, numdensgu2cgs) while the PR_KAPPAES hook uses the ko.h
//!    *macros* (kappaCGS2GU); both shapes preserved via thermo.Consts
//!    vs units.Units.
//!  * BOUNDELECTRON / BOUNDFREE / DOUBLECOMPTON / SUTHERLAND-DOPITA /
//!    CHIANTI are off in every target build and are not implemented.

const std = @import("std");
const simd = @import("../math/simd.zig");
const thermo = @import("thermo.zig");
const units_mod = @import("../units.zig");
const mesa = @import("mesa.zig");

/// C: struct opacities (the fields the M8 path fills), over lane type T;
/// the `<name>G` functions below are the comptime-T-generic cores of the
/// opacity chain (parallelization plan §2.2); the plain `<name>` scalar
/// API delegates to T = f64.
pub fn OpacOf(comptime T: type) type {
    return struct {
        gas_abs: T, // kappaGasAbs — Planck emission at T_e
        rad_abs: T, // kappaRadAbs — Planck absorption at T_r, T_e
        gas_num: T, // kappaGasNum
        rad_num: T, // kappaRadNum
        gas_ross: T, // kappaGasRoss
        rad_ross: T, // kappaRadRoss
        tot_emissivity: T,
    };
}
pub const Opac = OpacOf(f64);

/// Channel switches (comptime macros in C; runtime here; they only gate
/// whole terms). Defaults = the PUFFY build.
pub const Channels = struct {
    bremsstrahlung: bool = true,
    synchrotron: bool = true,
    kleinnishina: bool = true,
    comptonization: bool = true, // used by radforce.calcGi
    /// C: USE_SYNCHROTRON_BRIDGE_FUNCTIONS. The Ramesh NR bridge. Off = the
    /// Terelfactor suppression (the validated build); on = clamp Trad, add the
    /// NR synchrotron component, apply the number-opacity crossover, and drop
    /// the Terelfactor entirely (koral_lite_puffy).
    synchrotron_bridge: bool = false,
    /// C: MESA_KAPPA. When non-null, the free-free *Rosseland* opacity is
    /// replaced by this table's lookup (minus electron scattering); the
    /// free-free *Planck* absorption is still computed (as bremsstrahlung),
    /// regardless of the `bremsstrahlung` toggle, exactly as C's MESA branch.
    mesa: ?*const mesa.MesaTable = null,

    pub const puffy = Channels{};
};

/// Scalar inputs of calc_opacities_from_state, over lane type T.
pub fn StateInOf(comptime T: type) type {
    return struct {
        rho: T,
        tgas: T,
        te: T,
        trad: T,
        tradbb: T,
        ne: T,
        bsq: T,
    };
}
pub const StateIn = StateInOf(f64);

/// How many opacity channels a caller needs; a comptime channel-subset flag
/// so one formula body serves every path with bit-identical expression shapes
/// (finding #5). Every *consumed* channel keeps its exact expression, so the
/// narrower levels are bit-identical to `.full` on the fields they populate.
///   .full: every channel (golden/opacity tests, calc_kappa, tot_emissivity)
///   .residual; gas_abs + rad_abs + rad_ross (implicit residual / calcGiFromState);
///               drops rad_num, gas_ross, gas_ffross (two cbrt + two pow + one exp)
///   .chi: gas_abs only (wavespeed τ-limiter, radviscosity); also
///               drops every Trad-dependent channel (rad_abs/rad_ross and their
///               zeta/zbr machinery), since χ needs only κ = gas_abs
pub const OpacLevel = enum { full, residual, chi };

/// C: calc_opacities_from_state (opacities.c:186), BREMSSTRAHLUNG +
/// SYNCHROTRON channels, no SKIPFANCYOPACITIES. Returns kappa=kappaGasAbs
/// with all six opacity fields (tot_emissivity is filled by the caller,
/// calc_kappa_from_state: see calcKappaFromState).
pub fn calcOpacitiesFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) Opac {
    return calcOpacitiesFromStateG(f64, c, ch, s, .full);
}

/// calcOpacitiesFromState over lane type T. Scalar constants and the
/// all-scalar parens are broadcast whole (same f64 product as the scalar
/// code); the B = 0 Rosseland guard becomes a select with pow(0, −0.463)
/// computed-and-discarded on the guarded lanes.
///
/// `level` (comptime, see OpacLevel) selects the channel subset. `.residual`
/// drops rad_num / gas_ross / gas_ffross (two cbrt + two pow + one exp; the
/// implicit-residual hot path, radforce.zig:264); `.chi` also drops
/// every Trad-dependent channel (rad_abs / rad_ross and their zeta / zbr
/// machinery), leaving only κ = gas_abs for the wavespeed τ-limiter and the
/// radviscosity mfp (finding #5). Every *consumed* channel keeps its exact
/// expression, so each level is bit-identical to `.full` on the fields it
/// populates (gated in simd_tests.zig).
pub fn calcOpacitiesFromStateG(comptime T: type, c: *const thermo.Consts, ch: Channels, s: StateInOf(T), comptime level: OpacLevel) OpacOf(T) {
    const sp = simd.splat;
    const want_rad = level != .chi; // rad_abs / rad_ross channels
    const want_full = level == .full; // rad_num / gas_ross / gas_ffross
    const rho = s.rho;
    const te = s.te;
    const rt_te = @sqrt(te);

    const rhocgs = sp(T, c.rhogu2cgs) * rho;
    const nethcgs = s.ne * sp(T, c.numdensgu2cgs);

    // C: bsqcgs = fourmpi*endenGU2CGS(bsq) — exact M_PI and the ko.h macro
    const bsqcgs = sp(T, c.fourpi) * c.units.endenGu2CgsG(T, s.bsq);
    const bmagcgs = @sqrt(bsqcgs);

    const bbenergy = sp(T, 4.0 * c.sigma_rad_cgs) * te * te * te * te;

    var kappagasff: T = sp(T, 0);
    var kapparadff: T = sp(T, 0);
    var kappagasffross: T = sp(T, 0);
    var kapparadffross: T = sp(T, 0);
    var kappagassyn: T = sp(T, 0);
    var kapparadsyn: T = sp(T, 0);
    var kapparadnumsyn: T = sp(T, 0);
    var kapparadsynross: T = sp(T, 0);
    var kappagassynross: T = sp(T, 0);

    // Free-free (bremsstrahlung) Planck absorption + its Rosseland. C computes
    // the FF Planck terms in the plain BREMSSTRAHLUNG path AND, identically,
    // inside the MESA_KAPPA branch (opacities.c:308-316) — so MESA_KAPPA implies
    // the FF Planck absorption is on regardless of the BREMSSTRAHLUNG macro.
    // Only the FF *Rosseland* differs: with MESA it becomes a table lookup at
    // Trad (rad) / Te (gas), minus electron scattering, floored at 0.
    if (ch.bremsstrahlung or ch.mesa != null) {
        // free-free emissivity with Gaunt factor 1.2 and a relativistic
        // correction; n_avg = (X + Y + <Z²/A>·Z)·ρ/mp
        const emis_ff = sp(T, 1.4e-27) * nethcgs *
            sp(T, (c.comp.hfrac + c.comp.hefrac + c.comp.z2diva_mean * c.comp.mfrac()) / c.m_proton_cgs) *
            rt_te * sp(T, 1.2) * (sp(T, 1.0) + sp(T, 4.4e-10) * te);
        kappagasff = sp(T, c.kappacgs2gu) * (emis_ff / bbenergy) * rho;

        // electron-scattering opacity subtracted from the MESA Rosseland
        // (opacities.c:302, 0.2·(1+X)); only referenced on the MESA path.
        const kes_cgs = sp(T, 0.2 * (1.0 + c.comp.hfrac));

        // Trad-frame free-free absorption / Rosseland — dropped on the χ path
        // (κ = gas_abs carries no Trad term); the zeta ratios live only here.
        if (comptime want_rad) {
            const zeta = s.trad / te;
            const zeta_inv = sp(T, 1.0) / zeta;
            const zeta_inv_3 = zeta_inv * zeta_inv * zeta_inv;
            // no GRAY_BREMSS
            kapparadff = kappagasff * simd.log1p(T, sp(T, 1.6) * zeta) * sp(T, c.one_over_log_2p6) * zeta_inv_3;
            if (ch.mesa) |tab| {
                // MESA Rosseland at Trad (opacities.c:314-316)
                const k_rad_cgs = tab.lookupG(T, s.trad, rhocgs);
                kapparadffross = sp(T, c.kappacgs2gu) * @max(k_rad_cgs - kes_cgs, sp(T, 0.0)) * rho;
            } else {
                const zeta_root5 = simd.pow(T, zeta, 0.2);
                const zeta_root5_inv_4 = sp(T, 1.0) / (zeta_root5 * zeta_root5 * zeta_root5 * zeta_root5);
                const zeta_root5_inv_3 = zeta_root5 * zeta_root5_inv_4;
                const scale_pla_ros_ff = sp(T, 14.12) / (sp(T, 432.7) - sp(T, 106.8) * zeta_root5_inv_3 +
                    sp(T, 43.17) * zeta_root5_inv_4 + sp(T, 57.88) * zeta_inv);
                kapparadffross = kappagasff * scale_pla_ros_ff * zeta_inv_3;
            }
        }

        if (comptime want_full) {
            if (ch.mesa) |tab| {
                // MESA Rosseland at Te (opacities.c:305-307)
                const k_gas_cgs = tab.lookupG(T, te, rhocgs);
                kappagasffross = sp(T, c.kappacgs2gu) * @max(k_gas_cgs - kes_cgs, sp(T, 0.0)) * rho;
            } else {
                kappagasffross = kappagasff * sp(T, 0.0330); // gas_ross only
            }
        }
    }

    if (ch.synchrotron) {
        // C: USE_SYNCHROTRON_BRIDGE_FUNCTIONS (opacities.c:621-711). The Ramesh
        // NR bridge replaces the Terelfactor suppression with: (1) clamp the
        // synchrotron Trad up to Trad_lim = TradBB^(4/3)/Te^(1/3); (2) add an NR
        // component to the gas/rad synchrotron absorption; (3) apply a crossover
        // factor to the number opacity. Off ⇒ the validated Terelfactor path.
        const bridge = ch.synchrotron_bridge;
        const tc_n = sp(T, 5.07783e9); // number-opacity crossover temperature
        // Trad clamp (bridge only) — feeds every Trad-dependent synchrotron
        // channel (zbr). Non-bridge keeps s.trad, so zbr is bit-identical.
        const trad_syn: T = if (bridge) blk: {
            const trad_lim = simd.pow(T, s.tradbb, 4.0 / 3.0) / simd.pow(T, te, 1.0 / 3.0);
            break :blk simd.select(T, s.trad < trad_lim, trad_lim, s.trad);
        } else s.trad;

        const emis_synchro = sp(T, 3.61e-34) * (nethcgs / rhocgs) * te * te * bmagcgs * bmagcgs;
        kappagassyn = sp(T, c.kappacgs2gu) * (emis_synchro / bbenergy) * rho;

        // all Trad-frame synchrotron channels (rad_abs / rad_ross / rad_num)
        // are dropped on the χ path; only kappagassyn survives there.
        if (comptime want_rad) {
            const nu_mbsyn = sp(T, 1.19e-13) * te * te;
            const zbr = sp(T, c.k_boltz_cgs) * trad_syn / sp(T, c.h_cgs) / nu_mbsyn;

            const zeta_a_denom_rad =
                sp(T, 1.79) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs * bmagcgs * bmagcgs) +
                sp(T, 1.35) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
                sp(T, 0.248) * zbr * zbr * zbr;
            const ia_by_b_rad = bmagcgs * bmagcgs / zeta_a_denom_rad;

            const te5 = te * te * te * te * te;
            kapparadsyn = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_rad) * rho;

            // Bridge: NR component added to the rad synchrotron absorption
            // (opacities.c:665), Trad-based; dominant for Te ≲ 1e9 K.
            if (bridge) kapparadsyn += sp(T, c.kappacgs2gu) *
                (sp(T, 2.35869e-21) * (nethcgs / rhocgs) * (bmagcgs * bmagcgs) /
                    (trad_syn * trad_syn * trad_syn)) * rho;

            // number-of-photons averaged opacity (rad_num) — dead on the
            // implicit-residual / χ path (radforce.zig:264); full only.
            if (comptime want_full) {
                const zeta_a_denom_num =
                    (sp(T, 0.025) * simd.cbrt(T, zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
                        sp(T, 0.169) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * bmagcgs) +
                        sp(T, 0.287) * zbr * zbr);
                const ia_by_b_num = bmagcgs / zeta_a_denom_num;
                kapparadnumsyn = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_num) * rho;

                // Bridge: number-opacity crossover factor (opacities.c:680)
                if (bridge) kapparadnumsyn *= (te / tc_n) / (sp(T, 1.0) + te / tc_n);
            }

            // Rosseland; Bmb guards B = 0
            const bmb: T = simd.select(T, bmagcgs > sp(T, 0.0), simd.pow(T, bmagcgs, -0.463), sp(T, 0.0));
            const ia_by_b_ross_rad = sp(T, 0.13) * simd.pow(T, bmagcgs, 0.69) /
                (simd.pow(T, zbr, 1.69) * simd.exp(T, sp(T, 1.6) * simd.pow(T, zbr, 0.463) * bmb));
            kapparadsynross = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_ross_rad) * rho;

            // gas Rosseland (gas_ross) — dead on the same path; zbg feeds only
            // this channel (and reuses nu_mbsyn), so it lives here too.
            if (comptime want_full) {
                const zbg = sp(T, c.k_boltz_cgs) * te / sp(T, c.h_cgs) / nu_mbsyn;
                const ia_by_b_ross_gas = sp(T, 0.13) * simd.pow(T, bmagcgs, 0.69) /
                    (simd.pow(T, zbg, 1.69) * simd.exp(T, sp(T, 1.6) * simd.pow(T, zbg, 0.463) * bmb));
                kappagassynross = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_ross_gas) * rho;
            }
        }

        // Bridge: NR component added to the gas synchrotron absorption
        // (opacities.c:666), Te-based. Feeds gas_abs, so it applies on every
        // level (including the χ path — the only synchrotron term that survives
        // there). Placed after the Trad block; kappagassyn is otherwise untouched
        // by it, so the value matches C's earlier in-block add.
        if (bridge) kappagassyn += sp(T, c.kappacgs2gu) *
            (sp(T, 2.35869e-21) * (nethcgs / rhocgs) * (bmagcgs * bmagcgs) /
                (te * te * te)) * rho;

        // suppress synchrotron at nonrelativistic temperatures — the Terelfactor
        // path (opacities.c:700), used ONLY when the bridge is off. The bridge
        // replaces it with the NR adds / clamp / crossover above.
        if (!bridge) {
            const terel = te * sp(T, c.k_over_mecsq);
            const terelfactor = (terel * terel) / (sp(T, 1.0) + terel * terel);
            kappagassyn *= terelfactor;
            if (comptime want_rad) {
                kapparadsyn *= terelfactor;
                kapparadsynross *= terelfactor;
                if (comptime want_full) {
                    kapparadnumsyn *= terelfactor;
                    kappagassynross *= terelfactor;
                }
            }
        }
    }

    // sums (opacities.c:661-666); be/bf channels are off
    return .{
        .gas_abs = kappagasff + kappagassyn,
        .rad_abs = kapparadff + kapparadsyn,
        .gas_num = kappagasff, // synchrotron number opacity applied separately
        .rad_num = kapparadff + kapparadnumsyn,
        .gas_ross = kappagasffross + kappagassynross,
        .rad_ross = kapparadffross + kapparadsynross,
        .tot_emissivity = sp(T, 0), // filled by calcKappaFromState
    };
}

pub fn KappaResultOf(comptime T: type) type {
    return struct { kappa: T, opac: OpacOf(T) };
}
pub const KappaResult = KappaResultOf(f64);

/// C: calc_kappa_from_state (opacities.c:37). The default (no PR_KAPPA)
/// path plus the totEmissivity bookkeeping. Returns kappa = kappaGasAbs.
pub fn calcKappaFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) KappaResult {
    return calcKappaFromStateG(f64, c, ch, s, .full);
}

/// calcKappaFromState over lane type T. `level` (comptime) forwards to
/// calcOpacitiesFromStateG and, at `.full`, also fills
/// `tot_emissivity`, which no implicit-residual / χ consumer reads.
pub fn calcKappaFromStateG(comptime T: type, c: *const thermo.Consts, ch: Channels, s: StateInOf(T), comptime level: OpacLevel) KappaResultOf(T) {
    const sp = simd.splat;
    var opac = calcOpacitiesFromStateG(T, c, ch, s, level);
    if (comptime level == .full) {
        const b = sp(T, c.sigma_rad_over_pi) * s.te * s.te * s.te * s.te;
        // kappaGasAbs >= 0 always holds here, so totEmissivity uses it and no
        // field rewriting happens (opacities.c:80-99)
        opac.tot_emissivity = opac.gas_abs * sp(T, c.fourpi) * b;
    }
    return .{ .kappa = opac.gas_abs, .opac = opac };
}

/// C: PROBLEMS/PUFFY/kappaes.c via calc_kappaes_with_temperatures. The
/// KLEINNISHINA correction uses Tkn = Trad (which is Te on the standalone
/// calc_kappaes path and TradBB on the struct_of_state path). Uses the
/// ko.h kappaCGS2GU *macro* shape.
pub fn kappaEsPuffy(c: *const thermo.Consts, ch: Channels, rho: f64, trad: f64) f64 {
    return kappaEsPuffyG(f64, c, ch, rho, trad);
}

/// kappaEsPuffy over lane type T.
pub fn kappaEsPuffyG(comptime T: type, c: *const thermo.Consts, ch: Channels, rho: T, trad: T) T {
    const sp = simd.splat;
    var opac_correction: T = sp(T, 1.0);
    if (ch.kleinnishina) {
        const tkn = trad;
        opac_correction = sp(T, 1.0) / (sp(T, 1.0) + simd.pow(T, tkn / sp(T, 4.5e8), 0.86));
    }
    return c.units.kappaCgs2GuG(T, sp(T, 0.2 * (1.0 + c.comp.hfrac)) * opac_correction) * rho;
}

/// C: calc_tautot (opacities.c:159); τ = χ·dx per direction.
pub fn tautot(chi: f64, dx: [3]f64) [3]f64 {
    return .{ chi * dx[0], chi * dx[1], chi * dx[2] };
}

/// C: calc_tauabs (opacities.c:173).
pub fn tauabs(kappa: f64, dx: [3]f64) [3]f64 {
    return .{ kappa * dx[0], kappa * dx[1], kappa * dx[2] };
}
