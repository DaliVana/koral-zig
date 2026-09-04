//! `Core(cfg)`: the state every pass is allowed to see (redesign step 6,
//! 2026-09-04). The grid, the metric cache, the live primitive/conserved
//! pair, the per-cell flags and scalars, the physics and numerics options,
//! the worker team, the decomposition and comm backend, the timers, and
//! the CFL / order-reduction scalars the passes write. Plus the small
//! accessors and the MPI seam.
//!
//! A pass takes `*CoreT` and therefore cannot reach the integrator's stage
//! buffers, the face stores or another pass's scratch — ownership is
//! enforced by the type, not by a comment. `Sim(cfg)` (sim.zig) embeds a
//! Core as `.core`, adds the pass-owned scratch and the operators, and is
//! what problems, tests and the driver hold.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const field_mod = @import("../field.zig");
const metric = @import("../metric/metric.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const relele = @import("../relele.zig");
const p2u_mod = @import("../p2u.zig");
const threading = @import("../threading.zig");
const comm_mod = @import("../comm/comm.zig");
const storage = @import("storage.zig");
const bc = @import("bc.zig");
const polaraxis = @import("polaraxis.zig");
const timers_mod = @import("timers.zig");
const options = @import("options.zig");

const Grid = grid_mod.Grid;

pub const Error = relele.Error || error{OutOfMemory};

pub fn Core(comptime cfg: config.Config) type {
    comptime cfg.validate();
    const L = layout.VarLayout(cfg);
    const NV = L.count;

    return struct {
        const Self = @This();
        pub const Cfg = cfg;
        pub const Layout = L;
        pub const nv = NV;
        pub const FieldT = field_mod.Field(NV);
        /// C: MPI4CORNERS (force-defined by MAGNFIELD, choices.h:790): MHD
        /// sweeps reach ±1 ghost row and fill 2D corners.
        pub const wide = cfg.has(.mhd);
        /// Reconstruction order 0/1/2 from cfg.reconstruction.
        pub const base_order: u8 = switch (cfg.reconstruction) {
            .donor_cell => 0,
            .linear => 1,
            .ppm => 2,
        };
        pub const Flag = storage.Flag;
        pub const Scal = storage.Scal;

        allocator: std.mem.Allocator,
        grid: Grid,
        cache: precompute.MetricCache,
        p: FieldT,
        u: FieldT,
        scal: field_mod.Field(storage.n_scal),
        flags: []i32,
        phys: options.Physics,
        num: options.Numerics,
        /// P1: the persistent worker team all per-step passes dispatch on
        /// (threading.zig); null ≡ nthreads<=1 ≡ the serial path. Owned by
        /// the Sim.
        team: ?*threading.Team,
        /// The resolved decomposition (opt.decomp orelse trivial). x/y faces
        /// are always physical; z faces stop being boundaries when ntz > 1
        /// (their ghosts come from the exchange; sim/bc.zig gates on this).
        decomp: comm_mod.Decomp,
        comm: ?*comm_mod.Backend,
        /// P0 per-pass wall-clock instrumentation (sim/timers.zig); always
        /// accumulating: the driver prints/resets it at its output cadence.
        timers: timers_mod.PassTimers,
        tstepdenmax: f64,
        tstepdenmin: f64,
        min_dx: f64,
        min_dy: f64,
        min_dz: f64,
        /// C: REDUCEORDERATBH. The largest radial cell index whose center is
        /// inside the BL horizon (reconstruction order is dropped by one for
        /// cell[0] ≤ this). std.math.minInt when disabled / no cell is inside,
        /// so the `cell[0] <= …` test is always false. Set once in init (r is
        /// monotonic in the radial index, so a single threshold suffices).
        reduce_order_ix_max: i64,

        pub const Args = struct {
            phys: options.Physics,
            num: options.Numerics,
            team: ?*threading.Team,
            decomp: comm_mod.Decomp,
            comm: ?*comm_mod.Backend,
            /// The dynamo / radviscosity passes work in BL (OUTCOORDS): build
            /// the per-cell BL geometry sidecar and point the my2out Jacobian
            /// store at BL so both are precomputed once (finding #1).
            bl_cache: bool,
        };

        pub fn init(allocator: std.mem.Allocator, g: Grid, a: Args) !Self {
            var self: Self = undefined;
            self.allocator = allocator;
            self.grid = g;
            self.phys = a.phys;
            self.num = a.num;
            self.team = a.team;
            self.decomp = a.decomp;
            self.comm = a.comm;
            self.timers = .{};
            self.tstepdenmax = 0;
            self.tstepdenmin = storage.big;
            self.reduce_order_ix_max = std.math.minInt(i64);

            self.cache = try precompute.MetricCache.init(allocator, g, .{
                .coords = a.phys.coords,
                .out_coords = if (a.bl_cache) .bl else a.phys.coords,
                .mp = a.phys.mp,
                .bl_cache = a.bl_cache,
                // The team exists by now and the cache fills are pure
                // per-cell writes — bit-identical threaded.
                .team = a.team,
            });
            errdefer self.cache.deinit();

            // The big per-cell stores are allocated raw and zeroed THROUGH
            // THE TEAM (P4b NUMA first-touch): the first write to each page
            // then happens on a worker, spreading the pages across the
            // node's memory controllers. Zeros are zeros — bit-identical
            // serially (team == null degenerates to one whole-range memset).
            self.p = try FieldT.initUninitialized(allocator, g);
            errdefer self.p.deinit();
            threading.parallelZero(a.team, self.p.data);
            self.u = try FieldT.initUninitialized(allocator, g);
            errdefer self.u.deinit();
            threading.parallelZero(a.team, self.u.data);
            self.scal = try field_mod.Field(storage.n_scal).initUninitialized(allocator, g);
            errdefer self.scal.deinit();
            threading.parallelZero(a.team, self.scal.data);
            self.flags = try allocator.alloc(i32, g.cellCount() * storage.n_flags);
            errdefer allocator.free(self.flags);
            @memset(self.flags, 0);
            // MPI plan §6.2: the four persistent zero-copy channels bind
            // directly into p's storage (a heap slice; stable wherever the
            // Core itself lives).
            if (a.comm) |c| try c.bindExchange(self.p.data, g, NV);
            errdefer if (a.comm) |c| c.unbindExchange();

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
                        const dx = g.cellSize(ix, 0);
                        const dy = g.cellSize(iy, 1);
                        const dz = g.cellSize(iz, 2);
                        if (dx < mdx or mdx < 0) mdx = dx;
                        if (dy < mdx or mdy < 0) mdy = dy;
                        if (dz < mdx or mdz < 0) mdz = dz;
                    }
                }
            }
            self.min_dx = mdx;
            self.min_dy = mdy;
            self.min_dz = mdz;
            // Init fold (MPI plan §7.3): without it each rank seeds
            // initTimestepGuess from its own slab's minima and the first dt
            // diverges across ranks from step 0 (§1.1-5).
            if (a.comm) |c| {
                var fold = [3]f64{ self.min_dx, self.min_dy, self.min_dz };
                c.allreduceMin(fold[0..]);
                self.min_dx = fold[0];
                self.min_dy = fold[1];
                self.min_dz = fold[2];
            }

            // C: REDUCEORDERATBH threshold. The BL radius depends only on the
            // radial internal coordinate and increases monotonically with the
            // radial index, so the "inside the horizon" set is a prefix
            // [−ng, k) — record the largest such index once. Scan the full
            // radial extent incl. ghosts (the reconstruction stencil reads
            // them). θ is arbitrary for the r transform; use the domain mid-row.
            if (a.num.reduceorderatbh) {
                const r_h = metric.rHorizonBL(a.phys.mp.a);
                const y_mid = g.yc(@intCast(g.ny / 2));
                var jx: i64 = -@as(i64, @intCast(g.ng));
                while (jx < @as(i64, @intCast(g.nx + g.ng))) : (jx += 1) {
                    const xx = [4]f64{ 0, g.xc(jx), y_mid, 0 };
                    const rbl = coco.cocoN(xx, a.phys.coords, .bl, a.phys.mp)[1];
                    if (rbl < r_h) self.reduce_order_ix_max = jx;
                }
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.comm) |c| c.unbindExchange();
            self.p.deinit();
            self.u.deinit();
            self.scal.deinit();
            self.allocator.free(self.flags);
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
        pub fn nDim(self: *const Self, dim: usize) i64 {
            return switch (dim) {
                0 => self.nxi(),
                1 => self.nyi(),
                2 => self.nzi(),
                else => unreachable,
            };
        }

        // flagIdx/getFlag/setFlag/scGet/scSet are leaf accessors hit
        // O(cells × passes)/step; `inline` for Debug/ReleaseSafe (no FP change,
        // golden-safe — P2 #10).
        inline fn flagIdx(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) usize {
            // layout-independent cell index — was `p.cellOffset/NV`, which
            // coupled flag indexing to the primitives Field's AoS layout.
            return self.grid.cellIndex(ix, iy, iz) * storage.n_flags + @intFromEnum(f);
        }
        pub inline fn getFlag(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) i32 {
            return self.flags[self.flagIdx(f, ix, iy, iz)];
        }
        pub inline fn setFlag(self: *Self, f: Flag, ix: i64, iy: i64, iz: i64, v: i32) void {
            self.flags[self.flagIdx(f, ix, iy, iz)] = v;
        }

        pub inline fn scGet(self: *const Self, s: Scal, ix: i64, iy: i64, iz: i64) f64 {
            return self.scal.get(@intFromEnum(s), ix, iy, iz);
        }
        pub inline fn scSet(self: *Self, s: Scal, ix: i64, iy: i64, iz: i64, v: f64) void {
            self.scal.set(@intFromEnum(s), ix, iy, iz, v);
        }

        // ---- MPI seam (plan §5–§7) ----------------------------------------

        /// C: mpi_isitBC. Is this face a physical boundary of this rank's
        /// slab? Always true serially; only z faces can be interior (φ-only
        /// decomposition). sim/bc.zig gates every face fill on this.
        pub fn isPhysicalBoundary(self: *const Self, face: bc.BcFace) bool {
            return self.decomp.isPhysical(face);
        }

        /// One halo-exchange episode on the primitives (C: mpi_exchangedata).
        /// Zero-copy: Startall+Waitall over the persistent channels bound in
        /// init. No-op serially / at ntz==1. Main thread only (FUNNELED),
        /// which every call site satisfies (between team regions).
        pub fn exchangeHalos(self: *Self) void {
            if (self.comm) |c| {
                self.timers.begin(.halo);
                defer self.timers.end();
                c.exchange();
            }
        }

        /// Fold one scalar to the global max (identity serially). Init-time
        /// use: PUFFY's BETANORM (§1.1-5).
        pub fn globalMax(self: *Self, v: f64) f64 {
            if (self.comm) |c| {
                var buf = [1]f64{v};
                c.allreduceMax(buf[0..]);
                return buf[0];
            }
            return v;
        }

        /// Fold one scalar to the global sum (identity serially). Output-
        /// cadence use: diagnostic counters that must print as true totals,
        /// not per-rank values (plan §8.3 / review §10.2 n_radimp_fail).
        pub fn globalSum(self: *Self, v: f64) f64 {
            if (self.comm) |c| {
                var buf = [1]f64{v};
                c.allreduceSum(buf[0..]);
                return buf[0];
            }
            return v;
        }

        /// C: is_cell_corrected_polaraxis (finite.c:6132). The other half of
        /// doCorrect's contract: these rows are not evolved, they are
        /// overwritten. Derived from the same `polaraxis.band()`, so the
        /// predicate and the overwrite cannot claim different rows.
        pub fn isCellCorrectedPolaraxis(self: *const Self, iy: i64) bool {
            const b = polaraxis.band(Self, self) orelse return false;
            return b.owns(iy);
        }

        /// C: if_outsidegc. True for ghost *corner* cells (≥2 dims outside).
        pub fn isCorner(self: *const Self, ix: i64, iy: i64, iz: i64) bool {
            var n: u8 = 0;
            if (ix < 0 or ix >= self.nxi()) n += 1;
            if (iy < 0 or iy >= self.nyi()) n += 1;
            if (iz < 0 or iz >= self.nzi()) n += 1;
            return n >= 2;
        }

        /// Set one domain cell's primitives and derived conserveds
        /// (C: PR_INIT body; pp then p2u).
        pub fn initCell(self: *Self, ix: i64, iy: i64, iz: i64, pp_in: [NV]f64) Error!void {
            var pp = pp_in;
            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.phys.gam);
            self.p.store(ix, iy, iz, &pp);
            self.u.store(ix, iy, iz, &uu);
        }

        /// The CFL timestep from the last-computed wavespeeds: 1/tstepdenmax
        /// (C: 1/tstepdenmax). Single source of truth for the dt that step()
        /// uses and the driver recomputes; call it in both so they cannot
        /// drift. Requires the denominator to have been seeded
        /// (initTimestepGuess or a prior step); returns +inf if tstepdenmax is
        /// still 0, which the driver's pre-step guard rejects.
        pub fn cflDt(self: *const Self) f64 {
            return 1.0 / self.tstepdenmax;
        }
    };
}
