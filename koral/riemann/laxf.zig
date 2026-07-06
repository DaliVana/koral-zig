//! Approximate Riemann fluxes at a face from left/right-biased states
//! (C: f_calc_fluxes_at_faces, finite.c:1587/:1596).
//!
//! `ag` is the maximum |wavespeed| of the two adjacent cells; `al`/`ar`
//! the most negative / most positive characteristic speeds. Hydro and
//! radiation are combined as two separate systems by the caller (each row
//! family gets its own speeds).

/// Lax-Friedrichs: F★ = ½ (F_R + F_L − a_g (U_R − U_L)).
pub fn laxf(fl: f64, fr: f64, ul: f64, ur: f64, ag: f64) f64 {
    return 0.5 * (fr + fl - ag * (ur - ul));
}

/// HLL: upwind if the fan does not straddle the face, else the standard
/// combination (C: finite.c:1596-1611).
pub fn hll(fl: f64, fr: f64, ul: f64, ur: f64, al: f64, ar: f64) f64 {
    if (al > 0.0) return fl;
    if (ar < 0.0) return fr;
    return (-al * fr + ar * fl + al * ar * (ur - ul)) / (ar - al);
}
