//! Comptime-generated state-vector layout.
//!
//! Replaces the hand-maintained index arithmetic of KORAL's mnemonics.h
//! (`RHO=0 … ENTR=5, B1=NVHD+0, EE=NVMHD, …`) and the NV assembly of
//! choices.h. Variable order is exactly C's order so that dumps diff
//! index-for-index against the C code (architecture doc §4.1).

const std = @import("std");
const cfg_mod = @import("config.zig");
const Config = cfg_mod.Config;
const Module = cfg_mod.Module;

/// Every variable the state vector can carry. Nonthermal electron bins are
/// handled separately (they are a comptime-sized run, not individual tags).
pub const VarTag = enum {
    rho,
    uu,
    vx,
    vy,
    vz,
    entr,
    entre, // electron entropy (electrons module)
    entri, // ion entropy (electrons module)
    b1,
    b2,
    b3,
    uuff, // force-free block (forcefree module)
    vxff,
    vyff,
    vzff,
    ee, // radiation energy density
    fx, // radiation flux / rad-frame velocity primitives
    fy,
    fz,
    nf, // photon number (evolve_photon_number)
};

fn moduleTags(comptime cfg: Config, comptime m: Module) []const VarTag {
    return switch (m) {
        .hydro => &.{ .rho, .uu, .vx, .vy, .vz, .entr },
        .electrons => &.{ .entre, .entri },
        .relel => &.{}, // bins appended positionally, see VarLayout.count
        .mhd => &.{ .b1, .b2, .b3 },
        .forcefree => &.{ .uuff, .vxff, .vyff, .vzff },
        .radiation => if (cfg.evolve_photon_number)
            &.{ .ee, .fx, .fy, .fz, .nf }
        else
            &.{ .ee, .fx, .fy, .fz },
    };
}

pub fn VarLayout(comptime cfg: Config) type {
    comptime cfg.validate();

    // Build the ordered tag list. Relel bins occupy `n_relel_bins` slots
    // right after the electrons block (C: NEREL(i) = 8+i, inside NVHD).
    const built = comptime blk: {
        // The tag-list build can exceed the default 1000-branch quota when
        // its first evaluation is triggered from a deeply generic context
        // (e.g. a sampler method's parameter type); comptime-only, no
        // runtime effect.
        @setEvalBranchQuota(100_000);
        var tags_buf: []const VarTag = &.{};
        var offsets: [std.enums.values(VarTag).len]?usize = @splat(null);
        var n: usize = 0;
        var relel0: ?usize = null;
        for (cfg.modules) |m| {
            for (moduleTags(cfg, m)) |t| {
                offsets[@intFromEnum(t)] = n;
                tags_buf = tags_buf ++ &[_]VarTag{t};
                n += 1;
            }
            if (m == .relel) {
                relel0 = n;
                n += cfg.n_relel_bins;
            }
        }
        break :blk .{ .tags = tags_buf, .offsets = offsets, .count = n, .relel0 = relel0 };
    };

    return struct {
        /// Total number of evolved variables (C: NV).
        pub const count: usize = built.count;

        /// Tags in state-vector order (relel bins excluded. See relelIndex).
        pub const tags: []const VarTag = built.tags;

        /// Comptime index lookup: `index(.b1)` == C's B1. Compile error if the
        /// active module set does not provide the variable.
        pub fn index(comptime tag: VarTag) usize {
            return comptime built.offsets[@intFromEnum(tag)] orelse
                @compileError("variable ." ++ @tagName(tag) ++ " is not present in this Config");
        }

        /// True if the active module set carries this variable.
        pub fn hasVar(comptime tag: VarTag) bool {
            return comptime built.offsets[@intFromEnum(tag)] != null;
        }

        /// Index of nonthermal electron bin i (C: NEREL(i)).
        pub fn relelIndex(bin: usize) usize {
            const base = comptime (built.relel0 orelse
                @compileError("relelIndex: no .relel module in this Config"));
            std.debug.assert(bin < cfg.n_relel_bins);
            return base + bin;
        }
    };
}

//
// ---- M0 gate tests -------------------------------------------------------
//

test "PUFFY layout matches C mnemonics.h/choices.h exactly (NV=13)" {
    // C with MAGNFIELD + RADIATION, no electrons/photon-number:
    //   NVHD=6, NVMHD=9, NV=13
    //   RHO=0 UU=1 VX=2 VY=3 VZ=4 ENTR=5 B1=6 B2=7 B3=8 EE=9 FX=10 FY=11 FZ=12
    const L = VarLayout(cfg_mod.puffy);
    comptime {
        std.debug.assert(L.count == 13);
        std.debug.assert(L.index(.rho) == 0);
        std.debug.assert(L.index(.uu) == 1);
        std.debug.assert(L.index(.vx) == 2);
        std.debug.assert(L.index(.vy) == 3);
        std.debug.assert(L.index(.vz) == 4);
        std.debug.assert(L.index(.entr) == 5);
        std.debug.assert(L.index(.b1) == 6);
        std.debug.assert(L.index(.b2) == 7);
        std.debug.assert(L.index(.b3) == 8);
        std.debug.assert(L.index(.ee) == 9);
        std.debug.assert(L.index(.fx) == 10);
        std.debug.assert(L.index(.fy) == 11);
        std.debug.assert(L.index(.fz) == 12);
        std.debug.assert(!L.hasVar(.nf));
        std.debug.assert(!L.hasVar(.entre));
    }
}

test "hydro-only layout (C: no MAGNFIELD, no RADIATION → NV=NVHD=6)" {
    const L = VarLayout(.{ .modules = &.{.hydro} });
    comptime std.debug.assert(L.count == 6);
    comptime std.debug.assert(L.index(.entr) == 5);
}

test "two-temperature radiative MHD layout (C: EVOLVEELECTRONS → NVHD=8)" {
    // C: ENTRE=6, ENTRI=7, B1=NVHD+0=8, EE=NVMHD=11, NV=15
    const L = VarLayout(.{ .modules = &.{ .hydro, .electrons, .mhd, .radiation } });
    comptime {
        std.debug.assert(L.count == 15);
        std.debug.assert(L.index(.entre) == 6);
        std.debug.assert(L.index(.entri) == 7);
        std.debug.assert(L.index(.b1) == 8);
        std.debug.assert(L.index(.ee) == 11);
    }
}

test "photon number appends NF after FZ (C: NF=NVMHD+4)" {
    const L = VarLayout(.{
        .modules = &.{ .hydro, .mhd, .radiation },
        .evolve_photon_number = true,
    });
    comptime std.debug.assert(L.count == 14);
    comptime std.debug.assert(L.index(.nf) == 13);
}

test "relel bins sit inside the HD block before B (C: NVHD=6+2+NRELBIN)" {
    // C with RELELECTRONS, NRELBIN=4: NEREL(i)=8+i, B1=NVHD=12, NV(no rad,mhd)=15
    const L = VarLayout(.{
        .modules = &.{ .hydro, .electrons, .relel, .mhd },
        .n_relel_bins = 4,
    });
    comptime std.debug.assert(L.count == 15);
    try std.testing.expectEqual(@as(usize, 8), L.relelIndex(0));
    try std.testing.expectEqual(@as(usize, 11), L.relelIndex(3));
    comptime std.debug.assert(L.index(.b1) == 12);
}
