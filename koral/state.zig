//! Per-cell derived-state struct (C: struct_of_state, ko.h:437).
//!
//! M0 stub: only the base fluid fields exist. From M3 onward this becomes a
//! comptime composition — base fields merged with each active module's
//! `StateFields` (bcon/bsq/pmag from MHD; Ehat/Rij/Gi from radiation), so
//! only the fields the active physics needs are present.

const Config = @import("config.zig").Config;

pub fn State(comptime cfg: Config) type {
    comptime cfg.validate();
    return struct {
        rho: f64 = 0,
        uint: f64 = 0,
        gamma: f64 = 0, // Lorentz factor w.r.t. normal observer
        ucon: [4]f64 = @splat(0),
        ucov: [4]f64 = @splat(0),
        pgas: f64 = 0,
    };
}
