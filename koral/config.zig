//! Compile-time configuration: which physics modules exist, which algorithms
//! run, which coordinates the grid lives in. Everything here changes generated
//! code (variable layout, ghost depth, kernel selection) and is therefore
//! comptime; numbers a physicist tweaks between runs live in `params.zig`.
//!
//! Mirrors the roles of KORAL's `define.h`/`choices.h` macro layer
//! (see koral_lite/docs/zig-rewrite-architecture.md §2).

const std = @import("std");
const relele = @import("relele.zig");

/// Physics subsystems, in KORAL's fixed state-vector order:
/// hydro → electrons → relel → mhd → forcefree → radiation.
/// The order is C's order so a Zig dump diffs index-for-index against a C dump.
pub const Module = enum {
    hydro,
    electrons,
    relel,
    mhd,
    forcefree,
    radiation,
};

/// Reconstruction scheme (C: INT_ORDER 0/1/2 in avg2point, finite.c:97).
pub const Reconstruction = enum {
    donor_cell, // INT_ORDER 0
    linear, // INT_ORDER 1, minmod-theta limiter
    ppm, // INT_ORDER 2, Colella & Woodward

    pub fn ghostCells(self: Reconstruction) usize {
        return switch (self) {
            .donor_cell, .linear => 2,
            .ppm => 3,
        };
    }
};

/// Riemann flux combination (C: FLUXMETHOD, finite.c:1461).
pub const FluxMethod = enum { laxf, hll };

/// Time integrator (C: TIMESTEPPING, problem.c).
pub const TimeStepping = enum { rk1, rk2, rk2heun, rk2imex };

/// Coordinate system of the internal (uniform) grid (C: MYCOORDS).
pub const Coords = enum { mink, bl, ks, mks2 };

/// Velocity primitive parametrization (C: VELPRIM; VELR is the robust default).
/// Re-exports relele.VelType so there is a single VelType across the
/// codebase (its explicit 1/2/3 values are harmless here).
pub const VelType = relele.VelType;

pub const Config = struct {
    modules: []const Module,
    reconstruction: Reconstruction = .linear,
    flux: FluxMethod = .laxf,
    timestepping: TimeStepping = .rk2imex,
    coords: Coords = .mink,
    velprim: VelType = .velr,
    velprim_rad: VelType = .velr,
    /// C: EVOLVEPHOTONNUMBER — adds the NF variable to the radiation block.
    evolve_photon_number: bool = false,
    /// C: NRELBIN — number of nonthermal electron bins (relel module only).
    n_relel_bins: usize = 0,

    pub fn ghostCells(comptime self: Config) usize {
        return self.reconstruction.ghostCells();
    }

    pub fn has(comptime self: Config, comptime m: Module) bool {
        inline for (self.modules) |mod| {
            if (mod == m) return true;
        }
        return false;
    }

    /// Comptime sanity checks (C: am_i_sane()). Called by VarLayout/Simulation.
    pub fn validate(comptime self: Config) void {
        comptime {
            if (!self.has(.hydro))
                @compileError("koral.Config: the .hydro module is mandatory");
            if (self.has(.radiation) and !self.has(.hydro))
                @compileError("koral.Config: .radiation requires .hydro");
            if (self.has(.relel) and !self.has(.electrons))
                @compileError("koral.Config: .relel requires .electrons");
            if (self.has(.forcefree) and !self.has(.mhd))
                @compileError("koral.Config: .forcefree requires .mhd");
            if (self.has(.relel) and self.n_relel_bins == 0)
                @compileError("koral.Config: .relel requires n_relel_bins > 0");
            if (!self.has(.relel) and self.n_relel_bins != 0)
                @compileError("koral.Config: n_relel_bins set without .relel");
            if (self.evolve_photon_number and !self.has(.radiation))
                @compileError("koral.Config: evolve_photon_number requires .radiation");
            // Every call site hardcodes .velr; no other parametrization is
            // implemented yet, so catch a mis-set knob at compile time
            // instead of silently doing nothing.
            if (self.velprim != .velr)
                @compileError("koral.Config: only velprim = .velr is implemented");
            if (self.velprim_rad != .velr)
                @compileError("koral.Config: only velprim_rad = .velr is implemented");
            // Modules must appear in the canonical (C) order.
            var last: ?usize = null;
            for (self.modules) |m| {
                const ord = @intFromEnum(m);
                if (last) |l| {
                    if (ord <= l) @compileError(
                        "koral.Config: modules must be listed in canonical order " ++
                            "(hydro, electrons, relel, mhd, forcefree, radiation) without duplicates",
                    );
                }
                last = ord;
            }
        }
    }
};

/// The PUFFY configuration (PROBLEMS/PUFFY/define.h): MHD + M1 radiation,
/// single-temperature gas, no photon number, PPM, LAXF, RK2IMEX, MKS2.
pub const puffy = Config{
    .modules = &.{ .hydro, .mhd, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};

test "ghost cells follow reconstruction order (C: NG in ko.h)" {
    try std.testing.expectEqual(@as(usize, 3), puffy.ghostCells());
    const lin = Config{ .modules = &.{.hydro}, .reconstruction = .linear };
    try std.testing.expectEqual(@as(usize, 2), lin.ghostCells());
}

test "config validation accepts puffy" {
    comptime puffy.validate();
}
