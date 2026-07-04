//! The evolution driver — grid state, boundary conditions, the explicit
//! operator and the RK2IMEX step, transcribed from KORAL's serial path:
//!
//!   problem.c:141-402   RK2IMEX stage arithmetic (γ = 1 − 1/√2)
//!   finite.c:633        op_explicit (fused sweep: reconstruct → floors →
//!                       f_flux_prime per face side)
//!   finite.c:1461       f_calc_fluxes_at_faces (LAXF / HLL combination)
//!   finite.c:356/394    calc_wavespeeds / save_wavespeeds (timestep too)
//!   finite.c:546        calc_u2p (+ cell_fixup finite.c:5012)
//!   finite.c:2805       set_bc incl. the 2D corner filling (finite.c:3203)
//!   physics.c:789       f_metric_source_term_arb (GDETIN == 1 branch)
//!   u2p.c:2257          update_entropy
//!
//! C-fidelity notes:
//!  * "wide" sweep bounds mirror MPI4CORNERS, which choices.h:790 force-
//!    defines whenever MAGNFIELD is on — even in serial builds. So MHD
//!    configs sweep ±1 ghost rows and fill 2D ghost corners; hydro configs
//!    skip corners entirely (`if_outsidegc` continue).
//!  * All cell sizes/centers use C's per-cell FP forms (Grid.size/xc), and
//!    stage updates use C's exact `a*x + b*y` expression shapes so that
//!    forced-dt step tests can gate at 1e-13.
//!  * copyi_u (domain+ghosts, no corners) is replaced by full-array copies:
//!    the slots that differ (ghost corners of stage buffers) are never read.
//!  * op_implicit is a structural no-op until M9 (no RADIATION here);
//!    do_correct (polar axis) is a no-op until M11.
//!  * upreexplicit/ppreexplicit copies (finite.c:644) are skipped — nothing
//!    reads them before M12 (radviscosity / entropy mixing).

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const grid_mod = @import("grid.zig");
const field_mod = @import("field.zig");
const geometry = @import("geometry.zig");
const metric = @import("metric/metric.zig");
const precompute = @import("metric/precompute.zig");
const relele = @import("relele.zig");
const hydro = @import("physics/hydro.zig");
const flux_mod = @import("physics/flux.zig");
const wavespeeds = @import("physics/wavespeeds.zig");
const recon = @import("recon/recon.zig");
const invert = @import("solve/invert.zig");
const p2u_mod = @import("p2u.zig");
const laxf_mod = @import("flux/laxf.zig");
const ct = @import("magn/ct.zig");

const Grid = grid_mod.Grid;
const Geometry = geometry.Geometry;

pub const Error = relele.Error || error{OutOfMemory};

const big: f64 = 1.0e50; // C: BIG (ko.h) — only compared against, value moot

/// Boundary handling per axis (C: PERIODIC_?BC / COPY_?BC / SPECIFIC_BC).
pub const BcKind = enum { periodic, copy, specific };

/// Which boundary a ghost cell belongs to (C: XBCLO..ZBCHI).
pub const BcFace = enum { xlo, xhi, ylo, yhi, zlo, zhi };

/// Per-cell integer flags (C: cellflag; only the ones the M5/M6 path uses).
pub const Flag = enum(usize) {
    entropy, // ENTROPYFLAG — entropy inversion used this step
    entropy2, // ENTROPYFLAG2 — rad-energy borrowing failed (M7)
    entropy3, // ENTROPYFLAG3 — copy_entropycount snapshot
    hd_fixup, // HDFIXUPFLAG
    rad_fixup, // RADFIXUPFLAG (M7)
};
const n_flags = @typeInfo(Flag).@"enum".fields.len;

/// Cell-scalar slots (C: ahdxl..ahdz global arrays + cell_tstepden/cell_dt).
const Scal = enum(usize) {
    ahdxl,
    ahdxr,
    ahdyl,
    ahdyr,
    ahdzl,
    ahdzr,
    ahdx,
    ahdy,
    ahdz,
    tstepden,
    cell_dt,
};
const n_scal = @typeInfo(Scal).@"enum".fields.len;
const ahd_l = [3]Scal{ .ahdxl, .ahdyl, .ahdzl };
const ahd_r = [3]Scal{ .ahdxr, .ahdyr, .ahdzr };
const ahd_m = [3]Scal{ .ahdx, .ahdy, .ahdz };

/// Face-centered storage for one direction: like Field but with one extra
/// slice in its own dimension (C: ubx/uby/ubz arrays, set_ubx/get_ub).
pub fn FaceStore(comptime NV: usize) type {
    return struct {
        const Self = @This();
        data: []f64,
        nx_s: usize,
        ny_s: usize,
        nz_s: usize,
        ngx: i64,
        ngy: i64,
        ngz: i64,

        fn init(a: std.mem.Allocator, g: Grid, dim: usize) !Self {
            const nx_s = g.sx() + @intFromBool(dim == 0);
            const ny_s = g.sy() + @intFromBool(dim == 1);
            const nz_s = g.sz() + @intFromBool(dim == 2);
            const data = try a.alloc(f64, nx_s * ny_s * nz_s * NV);
            @memset(data, 0);
            return .{
                .data = data,
                .nx_s = nx_s,
                .ny_s = ny_s,
                .nz_s = nz_s,
                .ngx = @intCast(g.ngx),
                .ngy = @intCast(g.ngy),
                .ngz = @intCast(g.ngz),
            };
        }

        fn deinit(self: *Self, a: std.mem.Allocator) void {
            a.free(self.data);
        }

        fn offset(self: *const Self, ix: i64, iy: i64, iz: i64) usize {
            const jx: usize = @intCast(ix + self.ngx);
            const jy: usize = @intCast(iy + self.ngy);
            const jz: usize = @intCast(iz + self.ngz);
            std.debug.assert(jx < self.nx_s and jy < self.ny_s and jz < self.nz_s);
            return ((jz * self.ny_s + jy) * self.nx_s + jx) * NV;
        }

        pub fn get(self: *const Self, iv: usize, ix: i64, iy: i64, iz: i64) f64 {
            return self.data[self.offset(ix, iy, iz) + iv];
        }
        pub fn set(self: *Self, iv: usize, ix: i64, iy: i64, iz: i64, v: f64) void {
            self.data[self.offset(ix, iy, iz) + iv] = v;
        }
        pub fn load(self: *const Self, ix: i64, iy: i64, iz: i64, out: *[NV]f64) void {
            const off = self.offset(ix, iy, iz);
            @memcpy(out, self.data[off..][0..NV]);
        }
        pub fn store(self: *Self, ix: i64, iy: i64, iz: i64, pp: *const [NV]f64) void {
            const off = self.offset(ix, iy, iz);
            @memcpy(self.data[off..][0..NV], pp);
        }
        pub fn zero(self: *Self) void {
            @memset(self.data, 0);
        }
    };
}

pub fn Sim(comptime cfg: config.Config) type {
    comptime cfg.validate();
    const L = layout.VarLayout(cfg);
    const NV = L.count;
    // C: MPI4CORNERS is force-defined by MAGNFIELD (choices.h:790).
    const wide = cfg.has(.mhd);
    const base_order: u8 = switch (cfg.reconstruction) {
        .donor_cell => 0,
        .linear => 1,
        .ppm => 2,
    };

    return struct {
        const Self = @This();
        pub const Cfg = cfg;
        pub const Layout = L;
        pub const nv = NV;
        pub const FieldT = field_mod.Field(NV);
        pub const FaceT = FaceStore(NV);

        pub const SpecificBc = *const fn (
            sim: *const Self,
            ix: i64,
            iy: i64,
            iz: i64,
            t: f64,
            ifinit: bool,
            face: BcFace,
        ) [NV]f64;

        pub const Options = struct {
            coords: config.Coords,
            mp: metric.MetricParams = .{},
            gam: f64 = 5.0 / 3.0, // C: GAMMA
            tsteplim: f64 = 0.5, // C: TSTEPLIM
            minmod_theta: f64 = 1.5, // C: MINMOD_THETA
            floors: invert.FloorParams = invert.FloorParams.cdefault,
            do_fixups: bool = true, // C: DOFIXUPS && DOU2PMHDFIXUPS
            bc_x: BcKind = .copy,
            bc_y: BcKind = .copy,
            bc_z: BcKind = .copy,
            specific_bc: ?SpecificBc = null,
        };

        a: std.mem.Allocator,
        grid: Grid,
        cache: precompute.MetricCache,
        opt: Options,

        // state
        p: FieldT,
        u: FieldT,
        // RK2IMEX stage buffers (problem.c)
        ut0: FieldT,
        ut1: FieldT,
        ut2: FieldT,
        dut1: FieldT,
        dut2: FieldT,
        drt1: FieldT,
        drt2: FieldT,
        uforget: FieldT,
        ptm1: FieldT,
        ppostimplicit: FieldT,
        // fixup backups (finite.c:5030)
        u_bak: FieldT,
        p_bak: FieldT,

        // face-interpolated primitives & one-sided fluxes (pbLx.., flLx..)
        pb_l: [3]FaceT,
        pb_r: [3]FaceT,
        fl_l: [3]FaceT,
        fl_r: [3]FaceT,
        // combined fluxes (flbx/flby/flbz)
        flb: [3]FaceT,

        scal: field_mod.Field(n_scal),
        flags: []i32,

        // corner EMFs for flux-CT (comps 1..3 at slots 0..2), (nx+1)(ny+1)(nz+1)
        emf: [3][]f64,

        // vector-potential/B scratch for calc_BfromA (C: pvecpot — corner A
        // lives in slots B1..B3, the curl result in slots 1..3, coexisting).
        // Here: slots 0..2 = corner A_i, slots 3..5 = B^i.
        vecpot: field_mod.Field(6),

        t: f64 = 0,
        /// C: global_time — frozen at step start, used by set_bc.
        time: f64 = 0,
        dt: f64 = 0,
        /// dt the CFL logic would have chosen this step (before forcing).
        own_dt: f64 = 0,
        tstepdenmax: f64 = 0,
        tstepdenmin: f64 = big,
        min_dx: f64 = 0,
        min_dy: f64 = 0,
        min_dz: f64 = 0,
        nstep: u64 = 0,

        pub fn init(a: std.mem.Allocator, g: Grid, opt: Options) !Self {
            var self: Self = undefined;
            self.a = a;
            self.grid = g;
            self.opt = opt;
            self.t = 0;
            self.time = 0;
            self.dt = 0;
            self.own_dt = 0;
            self.tstepdenmax = 0;
            self.tstepdenmin = big;
            self.nstep = 0;

            self.cache = try precompute.MetricCache.init(a, g, .{
                .coords = opt.coords,
                .out_coords = opt.coords,
                .mp = opt.mp,
            });

            inline for (.{
                "p",           "u",    "ut0",   "ut1",     "ut2", "dut1", "dut2",
                "drt1",        "drt2", "uforget", "ptm1",
                "ppostimplicit", "u_bak", "p_bak",
            }) |name| {
                @field(self, name) = try FieldT.init(a, g);
            }
            for (0..3) |d| {
                self.pb_l[d] = try FaceT.init(a, g, d);
                self.pb_r[d] = try FaceT.init(a, g, d);
                self.fl_l[d] = try FaceT.init(a, g, d);
                self.fl_r[d] = try FaceT.init(a, g, d);
                self.flb[d] = try FaceT.init(a, g, d);
            }
            self.scal = try field_mod.Field(n_scal).init(a, g);
            self.flags = try a.alloc(i32, g.cellCount() * n_flags);
            @memset(self.flags, 0);
            const ncorn = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
            for (0..3) |c| {
                self.emf[c] = try a.alloc(f64, ncorn);
                @memset(self.emf[c], 0);
            }
            self.vecpot = try field_mod.Field(6).init(a, g);

            // C: set_grid's min-size scan (finite.c:1946-1961), including its
            // quirk of comparing dy/dz against *mdx*.
            var mdx: f64 = -1;
            var mdy: f64 = -1;
            var mdz: f64 = -1;
            var ix: i64 = 0;
            while (ix < self.nxi()) : (ix += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var iz: i64 = 0;
                    while (iz < self.nzi()) : (iz += 1) {
                        const dx = g.size(ix, 0);
                        const dy = g.size(iy, 1);
                        const dz = g.size(iz, 2);
                        if (dx < mdx or mdx < 0) mdx = dx;
                        if (dy < mdx or mdy < 0) mdy = dy;
                        if (dz < mdx or mdz < 0) mdz = dz;
                    }
                }
            }
            self.min_dx = mdx;
            self.min_dy = mdy;
            self.min_dz = mdz;

            return self;
        }

        pub fn deinit(self: *Self) void {
            const a = self.a;
            inline for (.{
                "p",           "u",    "ut0",   "ut1",     "ut2", "dut1", "dut2",
                "drt1",        "drt2", "uforget", "ptm1",
                "ppostimplicit", "u_bak", "p_bak",
            }) |name| {
                @field(self, name).deinit();
            }
            for (0..3) |d| {
                self.pb_l[d].deinit(a);
                self.pb_r[d].deinit(a);
                self.fl_l[d].deinit(a);
                self.fl_r[d].deinit(a);
                self.flb[d].deinit(a);
            }
            self.scal.deinit();
            a.free(self.flags);
            for (0..3) |c| a.free(self.emf[c]);
            self.vecpot.deinit();
            self.cache.deinit();
            self.* = undefined;
        }

        // ---- small helpers -------------------------------------------------

        pub fn nxi(self: *const Self) i64 {
            return @intCast(self.grid.nx);
        }
        pub fn nyi(self: *const Self) i64 {
            return @intCast(self.grid.ny);
        }
        pub fn nzi(self: *const Self) i64 {
            return @intCast(self.grid.nz);
        }
        fn nDim(self: *const Self, dim: usize) i64 {
            return switch (dim) {
                0 => self.nxi(),
                1 => self.nyi(),
                2 => self.nzi(),
                else => unreachable,
            };
        }

        fn flagIdx(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) usize {
            return (self.p.cellOffset(ix, iy, iz) / NV) * n_flags + @intFromEnum(f);
        }
        pub fn getFlag(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) i32 {
            return self.flags[self.flagIdx(f, ix, iy, iz)];
        }
        pub fn setFlag(self: *Self, f: Flag, ix: i64, iy: i64, iz: i64, v: i32) void {
            self.flags[self.flagIdx(f, ix, iy, iz)] = v;
        }

        fn scGet(self: *const Self, s: Scal, ix: i64, iy: i64, iz: i64) f64 {
            return self.scal.get(@intFromEnum(s), ix, iy, iz);
        }
        fn scSet(self: *Self, s: Scal, ix: i64, iy: i64, iz: i64, v: f64) void {
            self.scal.set(@intFromEnum(s), ix, iy, iz, v);
        }

        pub fn emfIdx(self: *const Self, ix: i64, iy: i64, iz: i64) usize {
            const g = &self.grid;
            const jx: usize = @intCast(ix);
            const jy: usize = @intCast(iy);
            const jz: usize = @intCast(iz);
            std.debug.assert(jx <= g.nx and jy <= g.ny and jz <= g.nz);
            return (jz * (g.ny + 1) + jy) * (g.nx + 1) + jx;
        }
        pub fn getEmf(self: *const Self, comp: usize, ix: i64, iy: i64, iz: i64) f64 {
            return self.emf[comp - 1][self.emfIdx(ix, iy, iz)];
        }
        pub fn setEmf(self: *Self, comp: usize, ix: i64, iy: i64, iz: i64, v: f64) void {
            self.emf[comp - 1][self.emfIdx(ix, iy, iz)] = v;
        }

        /// C: if_outsidegc — true for ghost *corner* cells (≥2 dims outside).
        fn isCorner(self: *const Self, ix: i64, iy: i64, iz: i64) bool {
            var n: u8 = 0;
            if (ix < 0 or ix >= self.nxi()) n += 1;
            if (iy < 0 or iy >= self.nyi()) n += 1;
            if (iz < 0 or iz >= self.nzi()) n += 1;
            return n >= 2;
        }

        // ---- initialization -------------------------------------------------

        /// Set one domain cell's primitives and derived conserveds
        /// (C: PR_INIT body — pp then p2u).
        pub fn initCell(self: *Self, ix: i64, iy: i64, iz: i64, pp_in: [NV]f64) Error!void {
            var pp = pp_in;
            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
            self.p.store(ix, iy, iz, &pp);
            self.u.store(ix, iy, iz, &uu);
        }

        /// ko.c init tail + problem.c:59-82: ghost fill and the initial
        /// timestep guess. Call after initCell over the domain (and after
        /// ct.calcBfromA for vector-potential problems).
        pub fn finishInit(self: *Self) Error!void {
            try self.setBc(0.0, true);
            self.initTimestepGuess();
        }

        /// problem.c:59-82 — initial dt guess from max_ws = 10⁴.
        pub fn initTimestepGuess(self: *Self) void {
            const ws: f64 = 10000.0;
            var tsd: f64 = undefined;
            if (self.grid.nz > 1) {
                tsd = ws / self.min_dx + ws / self.min_dy + ws / self.min_dz;
            } else if (self.grid.ny > 1) {
                tsd = ws / self.min_dx + ws / self.min_dy;
            } else {
                tsd = ws / self.min_dx;
            }
            tsd /= self.opt.tsteplim;
            self.tstepdenmax = tsd;
            self.tstepdenmin = tsd;

            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        self.scSet(.tstepden, ix, iy, iz, tsd);
                        self.scSet(.cell_dt, ix, iy, iz, 1.0 / tsd);
                    }
                }
            }
        }

        // ---- boundary conditions --------------------------------------------

        /// C: set_bc (finite.c:2805) — ghost cells (no corners), then for MHD
        /// ("MPI4CORNERS") builds the 2D corner surfaces + diagonals
        /// (finite.c:3203-3403; serial, so mpi_isitBC ≡ 1).
        pub fn setBc(self: *Self, t: f64, ifinit: bool) Error!void {
            const g = &self.grid;
            const nx = self.nxi();
            const ny = self.nyi();
            const nz = self.nzi();
            const ngx: i64 = @intCast(g.ngx);
            const ngy: i64 = @intCast(g.ngy);
            const ngz: i64 = @intCast(g.ngz);

            // x boundaries (ghost columns, domain rows)
            if (g.ngx > 0) {
                var iz: i64 = 0;
                while (iz < nz) : (iz += 1) {
                    var iy: i64 = 0;
                    while (iy < ny) : (iy += 1) {
                        var i: i64 = 1;
                        while (i <= ngx) : (i += 1) {
                            try self.setBcCell(-i, iy, iz, t, ifinit, .xlo);
                            try self.setBcCell(nx - 1 + i, iy, iz, t, ifinit, .xhi);
                        }
                    }
                }
            }
            // y boundaries
            if (g.ngy > 0) {
                var iz: i64 = 0;
                while (iz < nz) : (iz += 1) {
                    var ix: i64 = 0;
                    while (ix < nx) : (ix += 1) {
                        var i: i64 = 1;
                        while (i <= ngy) : (i += 1) {
                            try self.setBcCell(ix, -i, iz, t, ifinit, .ylo);
                            try self.setBcCell(ix, ny - 1 + i, iz, t, ifinit, .yhi);
                        }
                    }
                }
            }
            // z boundaries
            if (g.ngz > 0) {
                var iy: i64 = 0;
                while (iy < ny) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < nx) : (ix += 1) {
                        var i: i64 = 1;
                        while (i <= ngz) : (i += 1) {
                            try self.setBcCell(ix, iy, -i, t, ifinit, .zlo);
                            try self.setBcCell(ix, iy, nz - 1 + i, t, ifinit, .zhi);
                        }
                    }
                }
            }

            if (comptime wide) {
                if (g.ny > 1 and g.nz == 1) {
                    try self.fillCorners2d();
                } else if (g.nz > 1) {
                    // 3D / r-φ corner filling arrives with the problems that
                    // need it; no M5/M6 target is 3D.
                    @panic("Sim.setBc: 3D corner filling not implemented");
                }
            }
        }

        fn setBcCell(self: *Self, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: BcFace) Error!void {
            const kind: BcKind = switch (face) {
                .xlo, .xhi => self.opt.bc_x,
                .ylo, .yhi => self.opt.bc_y,
                .zlo, .zhi => self.opt.bc_z,
            };
            var pp: [NV]f64 = undefined;
            switch (kind) {
                .specific => {
                    pp = self.opt.specific_bc.?(self, ix, iy, iz, t, ifinit, face);
                },
                .periodic, .copy => {
                    var iix = ix;
                    var iiy = iy;
                    var iiz = iz;
                    const nx = self.nxi();
                    const ny = self.nyi();
                    const nz = self.nzi();
                    switch (face) {
                        .xlo, .xhi => switch (kind) {
                            .periodic => {
                                if (ix < 0) iix = ix + nx;
                                if (ix > nx - 1) iix = ix - nx;
                            },
                            .copy => {
                                if (ix < 0) iix = 0;
                                if (ix > nx - 1) iix = nx - 1;
                            },
                            else => unreachable,
                        },
                        .ylo, .yhi => switch (kind) {
                            .periodic => {
                                if (iy < 0) iiy = iy + ny;
                                if (iy > ny - 1) iiy = iy - ny;
                                // C quirk (finite.c:2756): NY<NG pins to 0
                                if (ny < @as(i64, @intCast(self.grid.ng))) iiy = 0;
                            },
                            .copy => {
                                if (iy < 0) iiy = 0;
                                if (iy > ny - 1) iiy = ny - 1;
                            },
                            else => unreachable,
                        },
                        .zlo, .zhi => switch (kind) {
                            .periodic => {
                                if (iz < 0) iiz = iz + nz;
                                if (iz > nz - 1) iiz = iz - nz;
                                if (nz < @as(i64, @intCast(self.grid.ng))) iiz = 0;
                            },
                            .copy => {
                                if (iz < 0) iiz = 0;
                                if (iz > nz - 1) iiz = nz - 1;
                            },
                            else => unreachable,
                        },
                    }
                    self.p.load(iix, iiy, iiz, &pp);
                },
            }
            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
            self.p.store(ix, iy, iz, &pp);
            self.u.store(ix, iy, iz, &uu);
        }

        /// p2u one ghost cell from its (already stored) primitives.
        fn p2uCell(self: *Self, ix: i64, iy: i64, iz: i64) Error!void {
            var pp: [NV]f64 = undefined;
            self.p.load(ix, iy, iz, &pp);
            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
            self.u.store(ix, iy, iz, &uu);
        }

        fn copyCellP(self: *Self, dix: i64, diy: i64, six: i64, siy: i64) void {
            var pp: [NV]f64 = undefined;
            self.p.load(six, siy, 0, &pp);
            self.p.store(dix, diy, 0, &pp);
        }

        fn avgCellP(self: *Self, dix: i64, diy: i64, ax: i64, ay: i64, bx: i64, by: i64) void {
            var pa: [NV]f64 = undefined;
            var pb: [NV]f64 = undefined;
            self.p.load(ax, ay, 0, &pa);
            self.p.load(bx, by, 0, &pb);
            var pp: [NV]f64 = undefined;
            for (0..NV) |iv| pp[iv] = 0.5 * (pa[iv] + pb[iv]);
            self.p.store(dix, diy, 0, &pp);
        }

        /// finite.c:3203-3403 — 2D (TNZ==1) total-corner filling: NG−1 deep
        /// one-cell surfaces copied from the adjacent domain row/column, then
        /// two diagonal cells averaged (periodic runs wrap the diagonals).
        fn fillCorners2d(self: *Self) Error!void {
            const nx = self.nxi();
            const ny = self.nyi();
            const ng: i64 = @intCast(self.grid.ng);
            const per_x = self.opt.bc_x == .periodic;
            const per_y = self.opt.bc_y == .periodic;

            // bottom-left
            {
                var i: i64 = 0;
                while (i < ng - 1) : (i += 1) {
                    self.copyCellP(-ng + i, -1, -ng + i, 0);
                    try self.p2uCell(-ng + i, -1, 0);
                    self.copyCellP(-1, -ng + i, 0, -ng + i);
                    try self.p2uCell(-1, -ng + i, 0);
                }
                var s1 = [4]i64{ -1, 0, 0, -1 }; // ix1,iy1,ix2,iy2
                if (per_y) s1 = .{ -1, ny - 1, -1, ny - 1 };
                if (per_x) s1 = .{ nx - 1, -1, nx - 1, -1 };
                self.avgCellP(-1, -1, s1[0], s1[1], s1[2], s1[3]);
                try self.p2uCell(-1, -1, 0);

                var s2 = [4]i64{ -2, -1, -1, -2 };
                if (per_y) s2 = .{ -2, ny - 2, -2, ny - 2 };
                if (per_x) s2 = .{ nx - 2, -2, nx - 2, -2 };
                self.avgCellP(-2, -2, s2[0], s2[1], s2[2], s2[3]);
                try self.p2uCell(-2, -2, 0);
            }
            // top-left
            {
                var i: i64 = 0;
                while (i < ng - 1) : (i += 1) {
                    self.copyCellP(-ng + i, ny, -ng + i, ny - 1);
                    try self.p2uCell(-ng + i, ny, 0);
                    self.copyCellP(-1, ny + i + 1, 0, ny + i + 1);
                    try self.p2uCell(-1, ny + i + 1, 0);
                }
                var s1 = [4]i64{ -1, ny - 1, 0, ny };
                if (per_y) s1 = .{ -1, 0, -1, 0 };
                if (per_x) s1 = .{ nx - 1, ny, nx - 1, ny };
                self.avgCellP(-1, ny, s1[0], s1[1], s1[2], s1[3]);
                try self.p2uCell(-1, ny, 0);

                var s2 = [4]i64{ -2, ny, -1, ny + 1 };
                if (per_y) s2 = .{ -2, 1, -2, 1 };
                if (per_x) s2 = .{ nx - 2, ny + 1, nx - 2, ny + 1 };
                self.avgCellP(-2, ny + 1, s2[0], s2[1], s2[2], s2[3]);
                try self.p2uCell(-2, ny + 1, 0);
            }
            // bottom-right
            {
                var i: i64 = 0;
                while (i < ng - 1) : (i += 1) {
                    self.copyCellP(nx + i + 1, -1, nx + i + 1, 0);
                    try self.p2uCell(nx + i + 1, -1, 0);
                    self.copyCellP(nx, -ng + i, nx - 1, -ng + i);
                    try self.p2uCell(nx, -ng + i, 0);
                }
                var s1 = [4]i64{ nx - 1, -1, nx, 0 };
                if (per_y) s1 = .{ nx, ny - 1, nx, ny - 1 };
                if (per_x) s1 = .{ 0, -1, 0, -1 };
                self.avgCellP(nx, -1, s1[0], s1[1], s1[2], s1[3]);
                try self.p2uCell(nx, -1, 0);

                var s2 = [4]i64{ nx, -2, nx + 1, -1 };
                if (per_y) s2 = .{ nx + 1, ny - 2, nx + 1, ny - 2 };
                if (per_x) s2 = .{ 1, -2, 1, -2 };
                self.avgCellP(nx + 1, -2, s2[0], s2[1], s2[2], s2[3]);
                try self.p2uCell(nx + 1, -2, 0);
            }
            // top-right
            {
                var i: i64 = 0;
                while (i < ng - 1) : (i += 1) {
                    self.copyCellP(nx + i + 1, ny, nx + i + 1, ny - 1);
                    try self.p2uCell(nx + i + 1, ny, 0);
                    self.copyCellP(nx, ny + i + 1, nx - 1, ny + i + 1);
                    try self.p2uCell(nx, ny + i + 1, 0);
                }
                var s1 = [4]i64{ nx - 1, ny, nx, ny - 1 };
                if (per_y) s1 = .{ nx, 0, nx, 0 };
                if (per_x) s1 = .{ 0, ny, 0, ny };
                self.avgCellP(nx, ny, s1[0], s1[1], s1[2], s1[3]);
                try self.p2uCell(nx, ny, 0);

                var s2 = [4]i64{ nx, ny + 1, nx + 1, ny };
                if (per_y) s2 = .{ nx + 1, 1, nx + 1, 1 };
                if (per_x) s2 = .{ 1, ny + 1, 1, ny + 1 };
                self.avgCellP(nx + 1, ny + 1, s2[0], s2[1], s2[2], s2[3]);
                try self.p2uCell(nx + 1, ny + 1, 0);
            }
        }

        // ---- wavespeeds & timestep -------------------------------------------

        /// C: calc_wavespeeds (finite.c:356) over domain + 1 ghost layer.
        pub fn calcWavespeeds(self: *Self) Error!void {
            const dims = [3]bool{ self.grid.nx > 1, self.grid.ny > 1, self.grid.nz > 1 };
            const lx: i64 = if (dims[0]) 1 else 0;
            const ly: i64 = if (dims[1]) 1 else 0;
            const lz: i64 = if (dims[2]) 1 else 0;

            var iz: i64 = -lz;
            while (iz < self.nzi() + lz) : (iz += 1) {
                var iy: i64 = -ly;
                while (iy < self.nyi() + ly) : (iy += 1) {
                    var ix: i64 = -lx;
                    while (ix < self.nxi() + lx) : (ix += 1) {
                        if (comptime !wide) {
                            if (self.isCorner(ix, iy, iz)) continue;
                        }
                        var pp: [NV]f64 = undefined;
                        self.p.load(ix, iy, iz, &pp);
                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const aaa = try wavespeeds.gasWavespeedsLr(cfg, pp, &geom, self.opt.gam, dims);
                        self.saveWavespeeds(ix, iy, iz, aaa);
                    }
                }
            }
        }

        /// C: save_wavespeeds (finite.c:394) — store per-cell speeds and,
        /// for domain cells, the CFL denominator (feeding the next dt).
        fn saveWavespeeds(self: *Self, ix: i64, iy: i64, iz: i64, aaa: [6]f64) void {
            for (0..3) |d| {
                self.scSet(ahd_l[d], ix, iy, iz, aaa[2 * d]);
                self.scSet(ahd_r[d], ix, iy, iz, aaa[2 * d + 1]);
            }
            const ax = @max(@abs(aaa[0]), @abs(aaa[1]));
            const ay = @max(@abs(aaa[2]), @abs(aaa[3]));
            const az = @max(@abs(aaa[4]), @abs(aaa[5]));
            self.scSet(.ahdx, ix, iy, iz, ax);
            self.scSet(.ahdy, ix, iy, iz, ay);
            self.scSet(.ahdz, ix, iy, iz, az);

            const in_domain = ix >= 0 and ix < self.nxi() and
                iy >= 0 and iy < self.nyi() and iz >= 0 and iz < self.nzi();
            if (!in_domain) return;

            const dx = self.grid.size(ix, 0);
            const dy = self.grid.size(iy, 1);
            const dz = self.grid.size(iz, 2);
            var tsd: f64 = undefined;
            if (self.grid.nz > 1 and self.grid.ny > 1) {
                tsd = ax / dx + ay / dy + az / dz;
            } else if (self.grid.nz == 1 and self.grid.ny > 1) {
                tsd = ax / dx + ay / dy;
            } else if (self.grid.ny == 1 and self.grid.nz > 1) {
                tsd = ax / dx + az / dz;
            } else {
                tsd = ax / dx;
            }
            tsd /= self.opt.tsteplim;
            self.scSet(.tstepden, ix, iy, iz, tsd);
            if (tsd > self.tstepdenmax) self.tstepdenmax = tsd;
            if (tsd < self.tstepdenmin) self.tstepdenmin = tsd;
        }

        /// C: save_timesteps (finite.c:490), no SHORTERTIMESTEP.
        fn saveTimesteps(self: *Self) void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        const tsd = self.scGet(.tstepden, ix, iy, iz);
                        self.scSet(.cell_dt, ix, iy, iz, 1.0 / tsd);
                    }
                }
            }
        }

        // ---- u2p over the grid ------------------------------------------------

        /// C: calc_u2p (finite.c:546) — per-cell inversion + floors, then
        /// fixup averaging and a boundary refresh.
        pub fn calcU2p(self: *Self) Error!void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        var uu: [NV]f64 = undefined;
                        var pp: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);
                        self.p.load(ix, iy, iz, &pp);

                        self.setFlag(.entropy, ix, iy, iz, 0);
                        self.setFlag(.entropy2, ix, iy, iz, 0);

                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const res = invert.u2pMhd(cfg, uu, &pp, &geom, self.opt.gam, self.opt.floors);
                        if (res.entropy_used) self.setFlag(.entropy, ix, iy, iz, 1);

                        _ = try invert.checkFloorsMhd(cfg, &pp, &geom, self.opt.gam, self.opt.floors);

                        self.p.store(ix, iy, iz, &pp);
                        self.setFlag(.hd_fixup, ix, iy, iz, if (res.fixup) 1 else 0);
                    }
                }
            }

            try self.cellFixup();
            try self.setBc(self.time, false);
        }

        /// C: cell_fixup(FIXUP_U2PMHD) (finite.c:5012). Averages flagged
        /// cells from their non-flagged in-domain neighbors; magnetic field
        /// and rho are never averaged (iv != RHO && iv < B1).
        fn cellFixup(self: *Self) Error!void {
            if (!self.opt.do_fixups) return;

            @memcpy(self.u_bak.data, self.u.data);
            @memcpy(self.p_bak.data, self.p.data);

            const b1_bound: usize = comptime if (L.hasVar(.b1)) L.index(.b1) else L.count;
            const nx = self.nxi();
            const ny = self.nyi();
            const nz = self.nzi();

            var iz: i64 = 0;
            while (iz < nz) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < ny) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < nx) : (ix += 1) {
                        if (self.getFlag(.hd_fixup, ix, iy, iz) == 0) continue;

                        var ppn: [6][NV]f64 = undefined;
                        var in_n: usize = 0;
                        // neighbor order matches C: x−, x+, y−, y+, z−, z+
                        const nbrs = [6][3]i64{
                            .{ ix - 1, iy, iz }, .{ ix + 1, iy, iz },
                            .{ ix, iy - 1, iz }, .{ ix, iy + 1, iz },
                            .{ ix, iy, iz - 1 }, .{ ix, iy, iz + 1 },
                        };
                        for (nbrs) |nb| {
                            if (nb[0] < 0 or nb[0] >= nx or nb[1] < 0 or nb[1] >= ny or
                                nb[2] < 0 or nb[2] >= nz) continue;
                            if (self.getFlag(.hd_fixup, nb[0], nb[1], nb[2]) != 0) continue;
                            self.p.load(nb[0], nb[1], nb[2], &ppn[in_n]);
                            in_n += 1;
                        }

                        const enough = (nz == 1 and ny == 1 and in_n >= 1) or
                            (nz == 1 and in_n >= 2) or
                            (ny == 1 and in_n >= 2) or
                            in_n >= 3;
                        if (!enough) continue; // C prints "didn't manage"

                        var pp: [NV]f64 = undefined;
                        self.p.load(ix, iy, iz, &pp);
                        for (0..NV) |iv| {
                            if (iv != L.index(.rho) and iv < b1_bound) {
                                var s: f64 = 0;
                                for (0..in_n) |k| s += ppn[k][iv];
                                pp[iv] = s / @as(f64, @floatFromInt(in_n));
                            }
                        }
                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
                        self.u_bak.store(ix, iy, iz, &uu);
                        self.p_bak.store(ix, iy, iz, &pp);
                    }
                }
            }

            @memcpy(self.u.data, self.u_bak.data);
            @memcpy(self.p.data, self.p_bak.data);
        }

        // ---- source terms -----------------------------------------------------

        /// C: f_metric_source_term_arb (physics.c:789), GDETIN == 1: only the
        /// Christoffel contraction rows survive; f_general_source_term is
        /// zero for every M5/M6 problem (no artificial heating/cooling).
        fn metricSource(self: *Self, ix: i64, iy: i64, iz: i64) Error![NV]f64 {
            const geom = self.cache.fillGeometry(ix, iy, iz);
            var pp: [NV]f64 = undefined;
            self.p.load(ix, iy, iz, &pp);

            const gdetu = geom.gdet;
            var ss: [NV]f64 = @splat(0);

            const tij = try hydro.calcTij(cfg, pp, &geom, self.opt.gam);
            const t = relele.indices2221(tij, &geom.gg);

            const rows = [4]usize{ L.index(.uu), L.index(.vx), L.index(.vy), L.index(.vz) };
            if (comptime L.hasVar(.ee)) {
                const radiation = @import("physics/radiation.zig");
                const rij_up = radiation.calcRij(cfg, pp, &geom);
                const rij = relele.indices2221(rij_up, &geom.gg);
                const rrows = [4]usize{ L.index(.ee), L.index(.fx), L.index(.fy), L.index(.fz) };
                for (0..4) |k| {
                    for (0..4) |l| {
                        for (0..4) |nu| {
                            ss[rows[nu]] += gdetu * t[k][l] * self.cache.kr(l, nu, k, ix, iy, iz);
                            ss[rrows[nu]] += gdetu * rij[k][l] * self.cache.kr(l, nu, k, ix, iy, iz);
                        }
                    }
                }
            } else {
                for (0..4) |k| {
                    for (0..4) |l| {
                        for (0..4) |nu| {
                            ss[rows[nu]] += gdetu * t[k][l] * self.cache.kr(l, nu, k, ix, iy, iz);
                        }
                    }
                }
            }
            return ss;
        }

        // ---- the explicit operator ---------------------------------------------

        /// Sweep bounds for the cross directions of a sweep/face loop:
        /// narrow = domain; wide (MHD) = ±1 where that dimension is active.
        fn crossLo(self: *const Self, dim: usize) i64 {
            if (comptime wide) {
                return if (self.nDim(dim) > 1) -1 else 0;
            }
            return 0;
        }
        fn crossHi(self: *const Self, dim: usize) i64 {
            if (comptime wide) {
                return if (self.nDim(dim) > 1) self.nDim(dim) + 1 else self.nDim(dim);
            }
            return self.nDim(dim);
        }

        /// One direction of op_explicit's interpolation sweep
        /// (finite.c:708-1181): reconstruct cell → face states, floor them,
        /// compute one-sided fluxes, stash into pb/fl face arrays.
        fn sweep(self: *Self, dim: usize) Error!void {
            const n = self.nDim(dim);
            if (n <= 1) return;
            const cross = switch (dim) {
                0 => [2]usize{ 1, 2 },
                1 => [2]usize{ 0, 2 },
                2 => [2]usize{ 0, 1 },
                else => unreachable,
            };

            var c1 = self.crossLo(cross[1]);
            while (c1 < self.crossHi(cross[1])) : (c1 += 1) {
                var c0 = self.crossLo(cross[0]);
                while (c0 < self.crossHi(cross[0])) : (c0 += 1) {
                    var i: i64 = -1;
                    while (i < n + 1) : (i += 1) {
                        var cell = [3]i64{ 0, 0, 0 };
                        cell[dim] = i;
                        cell[cross[0]] = c0;
                        cell[cross[1]] = c1;

                        const dol = i >= 0;
                        const dor = i < n;

                        const dx5 = [5]f64{
                            self.grid.size(i - 2, dim),
                            self.grid.size(i - 1, dim),
                            self.grid.size(i, dim),
                            self.grid.size(i + 1, dim),
                            self.grid.size(i + 2, dim),
                        };

                        var pm2: [NV]f64 = @splat(0);
                        var pm1: [NV]f64 = undefined;
                        var p0: [NV]f64 = undefined;
                        var pp1: [NV]f64 = undefined;
                        var pp2: [NV]f64 = @splat(0);
                        var c = cell;
                        c[dim] = i - 1;
                        self.p.load(c[0], c[1], c[2], &pm1);
                        c[dim] = i;
                        self.p.load(c[0], c[1], c[2], &p0);
                        c[dim] = i + 1;
                        self.p.load(c[0], c[1], c[2], &pp1);
                        // C guards the ±2 stencil with INT_ORDER>1 — with
                        // linear reconstruction there are only 2 ghost cells.
                        if (comptime base_order > 1) {
                            c[dim] = i - 2;
                            self.p.load(c[0], c[1], c[2], &pm2);
                            c[dim] = i + 2;
                            self.p.load(c[0], c[1], c[2], &pp2);
                        }

                        // REDUCEORDERWHENNEEDED / REDUCEMINMODTHETA are off
                        const f = recon.avg2point(NV, pm2, pm1, p0, pp1, pp2, dx5, base_order, 0, self.opt.minmod_theta);
                        var pl = f.ul;
                        var pr = f.ur;

                        var ffl: [NV]f64 = undefined;
                        var ffr: [NV]f64 = undefined;
                        if (dol) {
                            const geom = self.cache.fillGeometryFace(cell[0], cell[1], cell[2], dim);
                            _ = try invert.checkFloorsMhd(cfg, &pl, &geom, self.opt.gam, self.opt.floors);
                            ffl = try flux_mod.fFluxPrime(cfg, pl, dim, &geom, self.opt.gam);
                        }
                        if (dor) {
                            var cp = cell;
                            cp[dim] = i + 1;
                            const geom = self.cache.fillGeometryFace(cp[0], cp[1], cp[2], dim);
                            _ = try invert.checkFloorsMhd(cfg, &pr, &geom, self.opt.gam, self.opt.floors);
                            ffr = try flux_mod.fFluxPrime(cfg, pr, dim, &geom, self.opt.gam);
                        }

                        // pbR[dim] at face i ← pl; pbL[dim] at face i+1 ← pr
                        var cp = cell;
                        self.pb_r[dim].store(cp[0], cp[1], cp[2], &pl);
                        if (dol) self.fl_r[dim].store(cp[0], cp[1], cp[2], &ffl);
                        cp[dim] = i + 1;
                        self.pb_l[dim].store(cp[0], cp[1], cp[2], &pr);
                        if (dor) self.fl_l[dim].store(cp[0], cp[1], cp[2], &ffr);
                    }
                }
            }
        }

        /// C: f_calc_fluxes_at_faces (finite.c:1461) — for every face,
        /// combine the one-sided fluxes with LAXF or HLL using the saved
        /// cell wavespeeds. Hydro and radiation rows use separate speeds.
        fn fluxesAtFaces(self: *Self) Error!void {
            for (0..3) |d| self.flb[d].zero();

            for (0..3) |dim| {
                const n = self.nDim(dim);
                if (n <= 1) continue;
                const cross = switch (dim) {
                    0 => [2]usize{ 1, 2 },
                    1 => [2]usize{ 0, 2 },
                    2 => [2]usize{ 0, 1 },
                    else => unreachable,
                };

                var c1 = self.crossLo(cross[1]);
                while (c1 < self.crossHi(cross[1])) : (c1 += 1) {
                    var c0 = self.crossLo(cross[0]);
                    while (c0 < self.crossHi(cross[0])) : (c0 += 1) {
                        var i: i64 = 0;
                        while (i <= n) : (i += 1) {
                            var cf = [3]i64{ 0, 0, 0 };
                            cf[dim] = i;
                            cf[cross[0]] = c0;
                            cf[cross[1]] = c1;
                            var cm = cf;
                            cm[dim] = i - 1;

                            const ap1l = self.scGet(ahd_l[dim], cf[0], cf[1], cf[2]);
                            const ap1r = self.scGet(ahd_r[dim], cf[0], cf[1], cf[2]);
                            const ap1 = self.scGet(ahd_m[dim], cf[0], cf[1], cf[2]);
                            const am1l = self.scGet(ahd_l[dim], cm[0], cm[1], cm[2]);
                            const am1r = self.scGet(ahd_r[dim], cm[0], cm[1], cm[2]);
                            const am1 = self.scGet(ahd_m[dim], cm[0], cm[1], cm[2]);

                            var pLl: [NV]f64 = undefined;
                            var pRl: [NV]f64 = undefined;
                            self.pb_l[dim].load(cf[0], cf[1], cf[2], &pLl);
                            self.pb_r[dim].load(cf[0], cf[1], cf[2], &pRl);

                            const geom = self.cache.fillGeometryFace(cf[0], cf[1], cf[2], dim);
                            const uLl = try p2u_mod.p2u(cfg, pLl, &geom, self.opt.gam);
                            const uRl = try p2u_mod.p2u(cfg, pRl, &geom, self.opt.gam);

                            // hydro speeds; radiation rows get their own in M7
                            const ag = @max(ap1, am1);
                            const al = @min(ap1l, am1l);
                            const ar = @max(ap1r, am1r);

                            for (0..NV) |iv| {
                                const fl = self.fl_l[dim].get(iv, cf[0], cf[1], cf[2]);
                                const fr = self.fl_r[dim].get(iv, cf[0], cf[1], cf[2]);
                                const fstar = switch (comptime cfg.flux) {
                                    .laxf => laxf_mod.laxf(fl, fr, uLl[iv], uRl[iv], ag),
                                    .hll => laxf_mod.hll(fl, fr, uLl[iv], uRl[iv], al, ar),
                                };
                                self.flb[dim].set(iv, cf[0], cf[1], cf[2], fstar);
                            }
                        }
                    }
                }
            }
        }

        /// C: op_explicit (finite.c:633).
        pub fn opExplicit(self: *Self, t: f64, dtin: f64) Error!void {
            _ = t;
            // (upreexplicit/ppreexplicit copies skipped — see header)
            // calc_avgs_throughout: CALCHRONTHEGO only (M12)

            try self.calcWavespeeds();

            for (0..3) |dim| try self.sweep(dim);

            try self.fluxesAtFaces();

            if (comptime cfg.has(.mhd)) {
                ct.fluxCt(Self, self);
            }

            // conserved update: du from flux divergence + source terms
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        const ms = try self.metricSource(ix, iy, iz);
                        const dx = self.grid.size(ix, 0);
                        const dy = self.grid.size(iy, 1);
                        const dz = self.grid.size(iz, 2);
                        const dt = dtin;

                        for (0..NV) |iv| {
                            const flxl = self.flb[0].get(iv, ix, iy, iz);
                            const flxr = self.flb[0].get(iv, ix + 1, iy, iz);
                            const flyl = self.flb[1].get(iv, ix, iy, iz);
                            const flyr = self.flb[1].get(iv, ix, iy + 1, iz);
                            const flzl = self.flb[2].get(iv, ix, iy, iz);
                            const flzr = self.flb[2].get(iv, ix, iy, iz + 1);

                            const du = -(flxr - flxl) * dt / dx - (flyr - flyl) * dt / dy - (flzr - flzl) * dt / dz;
                            const val = self.u.get(iv, ix, iy, iz) + du + ms[iv] * dt;
                            self.u.set(iv, ix, iy, iz, val);
                        }
                    }
                }
            }

            // (EXPLICIT_LAB_RAD_SOURCE not used — PUFFY couples implicitly)

            try self.calcU2p();
            // (MIXENTROPIESPROPERLY off)
        }

        // ---- stage arithmetic (exact C expression shapes) ------------------------

        fn copyFull(dst: *FieldT, src: *const FieldT) void {
            @memcpy(dst.data, src.data);
        }

        /// dst = (1/Δ)·a + (−1/Δ)·b over the domain (problem.c:181/230/…).
        fn stageDeriv(self: *Self, dst: *FieldT, a_f: *const FieldT, b_f: *const FieldT, delta: f64) void {
            const f1 = 1.0 / delta;
            const f2 = -1.0 / delta;
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        for (0..NV) |iv| {
                            dst.set(iv, ix, iy, iz, f1 * a_f.get(iv, ix, iy, iz) + f2 * b_f.get(iv, ix, iy, iz));
                        }
                    }
                }
            }
        }

        /// dst = a + f1·b + f2·c over the domain (problem.c:243/357).
        fn stageCombine(self: *Self, dst: *FieldT, a_f: *const FieldT, f1: f64, b_f: *const FieldT, f2: f64, c_f: *const FieldT) void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        for (0..NV) |iv| {
                            dst.set(iv, ix, iy, iz, a_f.get(iv, ix, iy, iz) +
                                f1 * b_f.get(iv, ix, iy, iz) + f2 * c_f.get(iv, ix, iy, iz));
                        }
                    }
                }
            }
        }

        /// C: copy_entropycount (u2p.c:2237).
        fn copyEntropyCount(self: *Self) void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        self.setFlag(.entropy3, ix, iy, iz, self.getFlag(.entropy, ix, iy, iz));
                    }
                }
            }
        }

        /// C: update_entropy (u2p.c:2257) — recompute S(ρ,u) and refresh the
        /// MHD conserveds.
        pub fn updateEntropy(self: *Self) Error!void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < self.nyi()) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        var pp: [NV]f64 = undefined;
                        self.p.load(ix, iy, iz, &pp);
                        pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], self.opt.gam);
                        self.p.set(L.index(.entr), ix, iy, iz, pp[L.index(.entr)]);

                        var uu: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);
                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        try p2u_mod.p2uMhd(cfg, pp, &uu, &geom, self.opt.gam);
                        self.u.store(ix, iy, iz, &uu);
                    }
                }
            }
        }

        /// M9 will put the radiative implicit solve here; the stage
        /// arithmetic around it is already exact.
        fn opImplicit(self: *Self, t: f64, dtin: f64) void {
            _ = self;
            _ = t;
            _ = dtin;
        }

        /// M11: polar-axis correction (CORRECT_POLARAXIS). No-op until then.
        fn doCorrect(self: *Self) void {
            _ = self;
        }

        // ---- one full RK2IMEX step (problem.c:141-402) ----------------------------

        pub fn step(self: *Self, forced_dt: ?f64) Error!void {
            comptime std.debug.assert(cfg.timestepping == .rk2imex);

            const t = self.t;
            self.time = t; // C: global_time = t

            self.own_dt = 1.0 / self.tstepdenmax;
            const dt = forced_dt orelse self.own_dt;
            self.dt = dt;

            // reset wavespeed accumulators for this step
            self.tstepdenmax = -1;
            self.tstepdenmin = big;

            const gam_imex = 1.0 - 1.0 / @sqrt(2.0);

            // my_finger: PR_FINGER not defined for any target problem
            self.saveTimesteps();
            // set_gammagas: CONSISTENTGAMMA off

            // ---- 1st implicit ----
            copyFull(&self.ut0, &self.u);
            copyFull(&self.ptm1, &self.p);
            self.opImplicit(t, dt * gam_imex); // U(1) — no-op until M9
            copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt1, &self.u, &self.ut0, dt * gam_imex);

            // ---- 1st explicit ----
            copyFull(&self.ut1, &self.u);
            try self.calcU2p();
            self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(1))
            // apply_dynamo (M12), op_intermediate (electrons) — no-ops
            self.copyEntropyCount();
            self.stageDeriv(&self.dut1, &self.u, &self.ut1, dt);

            // ---- 1st together: u = ut0 + dt·dut1 + dt(1−2γ)·drt1 ----
            self.stageCombine(&self.u, &self.ut0, dt, &self.dut1, dt * (1.0 - 2.0 * gam_imex), &self.drt1);

            // ---- 2nd implicit ----
            copyFull(&self.uforget, &self.u);
            try self.calcU2p();
            copyFull(&self.ptm1, &self.p);
            self.doCorrect();
            self.opImplicit(t, gam_imex * dt); // U(2)
            copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt2, &self.u, &self.uforget, dt * gam_imex);

            // ---- 2nd explicit ----
            copyFull(&self.ut2, &self.u);
            try self.calcU2p();
            self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(2))
            self.stageDeriv(&self.dut2, &self.u, &self.ut2, dt);

            // ---- explicit together: u = ut0 + dt/2·(dut1 + dut2) ----
            self.stageCombine(&self.u, &self.ut0, dt / 2.0, &self.dut1, dt / 2.0, &self.dut2);
            // ---- implicit together: u += dt/2·(drt1 + drt2) ----
            self.stageCombine(&self.u, &self.u, dt / 2.0, &self.drt1, dt / 2.0, &self.drt2);

            // ---- final inversion & bookkeeping ----
            try self.calcU2p();
            self.doCorrect();
            try self.setBc(t, false);

            self.t = t + dt;
            try self.updateEntropy();
            self.nstep += 1;
        }
    };
}

test {
    _ = FaceStore(5);
}
