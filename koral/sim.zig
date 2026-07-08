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
//!  * All cell sizes/centers use C's per-cell FP forms (Grid.cellSize/xc), and
//!    stage updates use C's exact `a*x + b*y` expression shapes so that
//!    forced-dt step tests can gate at 1e-13.
//!  * copyi_u (domain+ghosts, no corners) is replaced by full-array copies:
//!    the slots that differ (ghost corners of stage buffers) are never read.
//!  * op_implicit (M9) is a structural no-op when opt.opac is null
//!    (≡ SKIPRADSOURCE); do_correct → correct_polaraxis (M10) is enabled
//!    by opt.correct_polaraxis (CORRECT_POLARAXIS, on for PUFFY in M11).
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
const invert_rad = @import("solve/invert_rad.zig");
const implicit = @import("solve/implicit.zig");
const radiation = @import("physics/radiation.zig");
const radforce = @import("physics/radforce.zig");
const radvisc_mod = @import("physics/radvisc.zig");
const p2u_mod = @import("p2u.zig");
const laxf_mod = @import("riemann/laxf.zig");
const ct = @import("magn/ct.zig");
const dynamo_mod = @import("magn/dynamo.zig");
const storage = @import("sim/storage.zig");
const bc = @import("sim/bc.zig");
const threading = @import("sim/threading.zig");
const timers_mod = @import("sim/timers.zig");

const Grid = grid_mod.Grid;
const Geometry = geometry.Geometry;

pub const Error = relele.Error || error{OutOfMemory};

const big: f64 = 1.0e50; // C: BIG (ko.h) — only compared against, value moot

// Storage & bookkeeping (face buffers, integer-flag / scalar-slot enums) live
// in sim/storage.zig; the boundary-condition enums + logic in sim/bc.zig;
// the per-pass wall-clock instrumentation (P0 of the parallelization plan)
// in sim/timers.zig. Re-exported here so external code keeps referencing
// them as sim.BcFace / sim.Flag / sim.FaceStore / sim.PassTimers.
pub const BcKind = bc.BcKind;
pub const BcFace = bc.BcFace;
pub const FaceStore = storage.FaceStore;
pub const Flag = storage.Flag;
pub const PassTimers = timers_mod.PassTimers;
/// Monotonic wall clock in ns (sim/timers.zig) — re-exported so the run
/// driver can time steps / throttle the heartbeat with the same clock.
pub const nowNs = timers_mod.nowNs;
const n_flags = storage.n_flags;
const Scal = storage.Scal;
const n_scal = storage.n_scal;
const ahd_l = storage.ahd_l;
const ahd_r = storage.ahd_r;
const ahd_m = storage.ahd_m;
const arad_l = storage.arad_l;
const arad_r = storage.arad_r;
const arad_m = storage.arad_m;

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

        /// A problem's boundary-condition hook. Returns the ghost-cell
        /// primitives, or an Error — the fallible frame/velocity conversions a
        /// boundary-adjacent cell can reach (SpacelikeVelocity,
        /// VelocityConversionFailed) propagate instead of being swallowed by
        /// `catch unreachable` (panic in safe builds, UB in ReleaseFast). The
        /// sole caller setBcCell already returns Error!void.
        pub const SpecificBc = *const fn (
            sim: *const Self,
            ix: i64,
            iy: i64,
            iz: i64,
            t: f64,
            ifinit: bool,
            face: BcFace,
        ) relele.Error![NV]f64;

        pub const Options = struct {
            coords: config.Coords,
            mp: metric.MetricParams = .{},
            gam: f64 = 5.0 / 3.0, // C: GAMMA
            tsteplim: f64 = 0.5, // C: TSTEPLIM
            minmod_theta: f64 = 1.5, // C: MINMOD_THETA
            floors: invert.FloorParams = invert.FloorParams.cdefault,
            rad: invert_rad.RadParams = invert_rad.RadParams.cdefault,
            /// Opacities for the rad wavespeed τ-limiter (C: calc_chi)
            /// AND the implicit radiative source coupling (op_implicit).
            /// null ≡ C's SKIPRADSOURCE: optically thin wavespeeds and no
            /// implicit operator.
            opac: ?radforce.Params = null,
            implicit: implicit.ImplicitParams = implicit.ImplicitParams.cdefault,
            do_fixups: bool = true, // C: DOFIXUPS && DOU2PMHDFIXUPS
            do_u2prad_fixups: bool = false, // C: DOU2PRADFIXUPS (0 everywhere)
            do_radimp_fixups: bool = false, // C: DORADIMPFIXUPS (0 everywhere)
            /// C: CORRECT_POLARAXIS (PUFFY define.h:107) — do_correct
            /// overwrites the most-polar rows and u2p/implicit/fixups skip
            /// them. Only meaningful on spherical-like coordinates.
            correct_polaraxis: bool = false,
            /// C: NCCORRECTPOLAR (PUFFY: 2).
            nccorrectpolar: i64 = 2,
            /// C: RADVISCOSITY==SHEARVISCOSITY (PUFFY define.h:96) — the
            /// radiative shear-viscosity contribution to R^ij, computed once
            /// per step (calc_Rij_visc_total) and added at the faces.
            radviscosity: bool = false,
            radvisc: radvisc_mod.Params = .{},
            /// C: MIMICDYNAMO (PUFFY define.h:66) — the mean-field dynamo run
            /// after each explicit sub-step (problem.c:210,326).
            dynamo: bool = false,
            dynamo_params: dynamo_mod.Params = .{},
            bc_x: BcKind = .copy,
            bc_y: BcKind = .copy,
            bc_z: BcKind = .copy,
            specific_bc: ?SpecificBc = null,
            /// Worker threads for the per-step passes (P1: all of them run
            /// on the persistent team). 1 ≡ the serial path (bit-identical;
            /// the golden tests run here).
            nthreads: usize = 1,
        };

        allocator: std.mem.Allocator,
        grid: Grid,
        cache: precompute.MetricCache,
        opt: Options,
        /// P1: the persistent worker team all per-step passes dispatch on
        /// (sim/threading.zig); null ≡ nthreads<=1 ≡ the serial path.
        team: ?*threading.Team,

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

        // C: Rijviscglobal — the per-cell viscous R^i_j (flattened i*4+j),
        // filled once per step over the domain + one non-corner ghost ring.
        rijvisc: field_mod.Field(16),

        // C: ptemp1 — the dynamo's cell-centered ΔA_φ scratch (slot 2 = φ);
        // and scaleth_otg — the per-radius density-weighted scale height.
        dyn_a: field_mod.Field(3),
        scaleth: []f64,

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
        /// implicit-solver diagnostics (C: global_int_slot counters), all
        /// run-cumulative — the driver deltas them for the per-step heartbeat.
        n_radimp_failures: u64 = 0,
        n_radimp_iters: u64 = 0,
        n_radimp_solves: u64 = 0,
        /// P0 per-pass wall-clock instrumentation (sim/timers.zig) — always
        /// accumulating; the driver prints/resets it at its output cadence.
        timers: timers_mod.PassTimers = .{},

        /// The full-Field members allocated in init and freed in deinit, in one
        /// place so the two lists cannot drift (P5). Startup allocation failure
        /// is treated as fatal — Sim.init has no per-alloc errdefer; a driver
        /// exits and the OS reclaims, and the test suite runs under the leak-
        /// checking allocator only after a *successful* init.
        const heap_field_names = .{
            "p",    "u",    "ut0",     "ut1",  "ut2",           "dut1",  "dut2",
            "drt1", "drt2", "uforget", "ptm1", "ppostimplicit", "u_bak", "p_bak",
        };

        pub fn init(allocator: std.mem.Allocator, g: Grid, opt: Options) !Self {
            // Runtime precondition checks (P1 correctness). Each unchecked
            // invariant otherwise fails far from its cause; we return
            // error.InvalidConfig rather than std.debug.assert so ReleaseFast
            // production builds are covered too. Cost is negligible (once/run).
            //
            // (1) Ghost depth must cover the reconstruction stencil. PPM's
            //     unconditional i−2 load (with the ±1 MHD cross range) drives
            //     Field.cellOffset's @intCast negative when g.ng is too small —
            //     a safe-build panic, OOB UB in ReleaseFast — deep in the sweep.
            if (g.ng < cfg.ghostCells()) return error.InvalidConfig;
            // (2) A `.specific` boundary needs its callback, else setBcCell
            //     unwraps a null specific_bc on the first ghost fill.
            if ((opt.bc_x == .specific or opt.bc_y == .specific or opt.bc_z == .specific) and
                opt.specific_bc == null) return error.InvalidConfig;
            // (3) The runtime metric coords (drives the MetricCache) must match
            //     the comptime coords every other consumer reads via Cfg.coords
            //     (dynamo/scalars/puffy coco transforms); divergence silently
            //     mixes two coordinate systems — wrong physics, no diagnostic.
            if (opt.coords != cfg.coords) return error.InvalidConfig;
            // (4) Radiative shear viscosity needs opacities: ν = α·mfp, mfp =
            //     1/χ, and χ = calc_chi is undefined without opac. C's
            //     SKIPRADSOURCE keeps viscosity active via mfp=1e50, but the Zig
            //     path would silently store ν=0 with the sweep still running.
            //     Reject the unsupported combination outright.
            if (comptime L.hasVar(.ee)) {
                if (opt.radviscosity and opt.opac == null) return error.InvalidConfig;
            }
            // (5) correct_polaraxis overwrites the most-polar nccorrectpolar rows
            //     per side; with ny ≤ 2·nccorrectpolar every row is "corrected"
            //     and the overwrite sources lie inside the overwritten band —
            //     order-dependent garbage (the module supports reduced ny for
            //     tests, so this is reachable).
            if (opt.correct_polaraxis and
                @as(i64, @intCast(g.ny)) <= 2 * opt.nccorrectpolar) return error.InvalidConfig;

            var self: Self = undefined;
            self.allocator = allocator;
            self.grid = g;
            self.opt = opt;
            self.t = 0;
            self.time = 0;
            self.dt = 0;
            self.own_dt = 0;
            self.tstepdenmax = 0;
            self.tstepdenmin = big;
            self.nstep = 0;
            self.n_radimp_failures = 0;
            self.n_radimp_iters = 0;
            self.n_radimp_solves = 0;
            self.timers = .{};
            self.team = if (opt.nthreads > 1) try threading.Team.init(allocator, opt.nthreads) else null;

            // The dynamo / radviscosity passes work in BL (OUTCOORDS): build
            // the per-cell BL geometry sidecar and point the my2out Jacobian
            // store at BL so both are precomputed once instead of per sub-step
            // per cell (finding #1). Kerr coords are guaranteed here — both
            // switches imply a Kerr problem.
            const want_bl = opt.dynamo or opt.radviscosity;
            self.cache = try precompute.MetricCache.init(allocator, g, .{
                .coords = opt.coords,
                .out_coords = if (want_bl) .bl else opt.coords,
                .mp = opt.mp,
                .bl_cache = want_bl,
            });

            inline for (heap_field_names) |name| {
                @field(self, name) = try FieldT.init(allocator, g);
            }
            for (0..3) |d| {
                self.pb_l[d] = try FaceT.init(allocator, g, d);
                self.pb_r[d] = try FaceT.init(allocator, g, d);
                self.fl_l[d] = try FaceT.init(allocator, g, d);
                self.fl_r[d] = try FaceT.init(allocator, g, d);
                self.flb[d] = try FaceT.init(allocator, g, d);
            }
            self.scal = try field_mod.Field(n_scal).init(allocator, g);
            self.flags = try allocator.alloc(i32, g.cellCount() * n_flags);
            @memset(self.flags, 0);
            const ncorn = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
            for (0..3) |c| {
                self.emf[c] = try allocator.alloc(f64, ncorn);
                @memset(self.emf[c], 0);
            }
            self.vecpot = try field_mod.Field(6).init(allocator, g);
            self.rijvisc = try field_mod.Field(16).init(allocator, g);
            self.dyn_a = try field_mod.Field(3).init(allocator, g);
            self.scaleth = try allocator.alloc(f64, g.nx);
            @memset(self.scaleth, 0);

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

            return self;
        }

        pub fn deinit(self: *Self) void {
            const allocator = self.allocator;
            if (self.team) |tm| tm.deinit();
            inline for (heap_field_names) |name| {
                @field(self, name).deinit();
            }
            for (0..3) |d| {
                self.pb_l[d].deinit();
                self.pb_r[d].deinit();
                self.fl_l[d].deinit();
                self.fl_r[d].deinit();
                self.flb[d].deinit();
            }
            self.scal.deinit();
            allocator.free(self.flags);
            for (0..3) |c| allocator.free(self.emf[c]);
            self.vecpot.deinit();
            self.rijvisc.deinit();
            self.dyn_a.deinit();
            allocator.free(self.scaleth);
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
            return (self.p.cellOffset(ix, iy, iz) / NV) * n_flags + @intFromEnum(f);
        }
        pub inline fn getFlag(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) i32 {
            return self.flags[self.flagIdx(f, ix, iy, iz)];
        }
        pub inline fn setFlag(self: *Self, f: Flag, ix: i64, iy: i64, iz: i64, v: i32) void {
            self.flags[self.flagIdx(f, ix, iy, iz)] = v;
        }

        inline fn scGet(self: *const Self, s: Scal, ix: i64, iy: i64, iz: i64) f64 {
            return self.scal.get(@intFromEnum(s), ix, iy, iz);
        }
        inline fn scSet(self: *Self, s: Scal, ix: i64, iy: i64, iz: i64, v: f64) void {
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

        /// C: is_cell_corrected_polaraxis (finite.c:6132) — within
        /// NCCORRECTPOLAR of either θ-edge (no HALFTHETA; serial owns both
        /// tiles). Note C does NOT check the coordinate system here — only
        /// correct_polaraxis' overwrite does.
        fn isCellCorrectedPolaraxis(self: *const Self, iy: i64) bool {
            if (!self.opt.correct_polaraxis) return false;
            return iy < self.opt.nccorrectpolar or
                iy > self.nyi() - self.opt.nccorrectpolar - 1;
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

        /// The CFL timestep from the last-computed wavespeeds: 1/tstepdenmax
        /// (C: 1/tstepdenmax). Single source of truth for the dt that step()
        /// uses and the driver recomputes — call it in both so they cannot
        /// drift. Requires the denominator to have been seeded
        /// (initTimestepGuess or a prior step); returns +inf if tstepdenmax is
        /// still 0, which the driver's pre-step guard rejects.
        pub fn cflDt(self: *const Self) f64 {
            return 1.0 / self.tstepdenmax;
        }

        // ---- boundary conditions --------------------------------------------

        /// C: set_bc (finite.c:2805). The implementation (ghost fill + the 2D
        /// corner filling, and the MPI-growth seam) lives in sim/bc.zig; this
        /// is the stable method entry point that problems/tests call.
        pub fn setBc(self: *Self, t: f64, ifinit: bool) Error!void {
            self.timers.begin(.bc);
            defer self.timers.end();
            return bc.setBc(Self, self, t, ifinit);
        }

        // ---- wavespeeds & timestep -------------------------------------------

        /// C: calc_wavespeeds (finite.c:356) over domain + 1 ghost layer.
        /// Band-parallel over iy; the CFL-denominator max/min reduce runs as
        /// per-worker partials merged after the region (max/min are
        /// order-insensitive, so this is bit-identical to the serial scan).
        pub fn calcWavespeeds(self: *Self) Error!void {
            self.timers.begin(.wavespeeds);
            defer self.timers.end();
            const ly: i64 = if (self.grid.ny > 1) 1 else 0;
            const res = threading.parallelRange(Self, self, self.team, -ly, self.nyi() + ly, wavespeedRowsWorker);
            if (res.err) |e| return e;
            if (res.tsd_max > self.tstepdenmax) self.tstepdenmax = res.tsd_max;
            if (res.tsd_min < self.tstepdenmin) self.tstepdenmin = res.tsd_min;
        }

        fn wavespeedRowsWorker(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            self.wavespeedRows(iy0, iy1, res) catch |e| {
                res.err = e;
            };
        }

        /// The per-cell wavespeed body for iy ∈ [iy0, iy1) (all iz, all ix
        /// incl. the ±1 ghost layer).
        fn wavespeedRows(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) Error!void {
            const active_dims = [3]bool{ self.grid.nx > 1, self.grid.ny > 1, self.grid.nz > 1 };
            const lx: i64 = if (active_dims[0]) 1 else 0;
            const lz: i64 = if (active_dims[2]) 1 else 0;

            var iz: i64 = -lz;
            while (iz < self.nzi() + lz) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = -lx;
                    while (ix < self.nxi() + lx) : (ix += 1) {
                        if (comptime !wide) {
                            if (self.isCorner(ix, iy, iz)) continue;
                        }
                        var pp: [NV]f64 = undefined;
                        self.p.load(ix, iy, iz, &pp);
                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const aaa = try wavespeeds.gasWavespeedsLr(cfg, pp, &geom, self.opt.gam, active_dims);

                        var rad0: [6]f64 = @splat(0);
                        var radl: [6]f64 = @splat(0);
                        if (comptime L.hasVar(.ee)) {
                            // optical depth over the cell (physics.c:549-558);
                            // ix could be a face index in C, hence the max of
                            // left/right sizes
                            var tautot: [3]f64 = @splat(0);
                            if (self.opt.opac) |*op| {
                                const chi = try radforce.calcChiSlim(cfg, pp, &geom, self.opt.gam, op);
                                const idx3 = [3]i64{ ix, iy, iz };
                                for (0..3) |d| {
                                    const sg = @sqrt(geom.gg[d + 1][d + 1]);
                                    const dxp = @max(
                                        self.grid.cellSize(idx3[d], d) * sg,
                                        self.grid.cellSize(idx3[d] + 1, d) * sg,
                                    );
                                    tautot[d] = chi * dxp;
                                }
                            }
                            const aval = try radiation.calcRadWavespeeds(cfg, pp, &geom, tautot, active_dims);
                            for (0..6) |i| {
                                rad0[i] = aval[i];
                                radl[i] = aval[6 + i];
                            }
                            // zero 'co-going' velocities (physics.c:609-621)
                            for (0..3) |d| {
                                if (radl[2 * d] > 0.0) radl[2 * d] = 0.0;
                                if (radl[2 * d + 1] < 0.0) radl[2 * d + 1] = 0.0;
                                if (rad0[2 * d] > 0.0) rad0[2 * d] = 0.0;
                                if (rad0[2 * d + 1] < 0.0) rad0[2 * d + 1] = 0.0;
                            }
                        }
                        self.saveWavespeeds(ix, iy, iz, aaa, rad0, radl, res);
                    }
                }
            }
        }

        /// C: save_wavespeeds (finite.c:394) — store per-cell speeds and,
        /// for domain cells, the CFL denominator (feeding the next dt).
        /// rad0 (unlimited by τ) only enters the timestep; radl (τ-limited)
        /// is what the flux combination reads. The dt max/min go into the
        /// worker's ChunkResult partials (merged after the region).
        fn saveWavespeeds(self: *Self, ix: i64, iy: i64, iz: i64, aaa: [6]f64, rad0: [6]f64, radl: [6]f64, res: *threading.ChunkResult) void {
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

            var wsx = ax;
            var wsy = ay;
            var wsz = az;
            if (comptime L.hasVar(.ee)) {
                for (0..3) |d| {
                    self.scSet(arad_l[d], ix, iy, iz, radl[2 * d]);
                    self.scSet(arad_r[d], ix, iy, iz, radl[2 * d + 1]);
                    self.scSet(arad_m[d], ix, iy, iz, @max(@abs(radl[2 * d]), @abs(radl[2 * d + 1])));
                }
                // timestep uses the wavespeeds NOT limited by optical depth
                wsx = @max(ax, @max(@abs(rad0[0]), @abs(rad0[1])));
                wsy = @max(ay, @max(@abs(rad0[2]), @abs(rad0[3])));
                wsz = @max(az, @max(@abs(rad0[4]), @abs(rad0[5])));
            }

            const in_domain = ix >= 0 and ix < self.nxi() and
                iy >= 0 and iy < self.nyi() and iz >= 0 and iz < self.nzi();
            if (!in_domain) return;

            const dx = self.grid.cellSize(ix, 0);
            const dy = self.grid.cellSize(iy, 1);
            const dz = self.grid.cellSize(iz, 2);
            var tsd: f64 = undefined;
            if (self.grid.nz > 1 and self.grid.ny > 1) {
                tsd = wsx / dx + wsy / dy + wsz / dz;
            } else if (self.grid.nz == 1 and self.grid.ny > 1) {
                tsd = wsx / dx + wsy / dy;
            } else if (self.grid.ny == 1 and self.grid.nz > 1) {
                tsd = wsx / dx + wsz / dz;
            } else {
                tsd = wsx / dx;
            }
            tsd /= self.opt.tsteplim;
            self.scSet(.tstepden, ix, iy, iz, tsd);
            if (tsd > res.tsd_max) res.tsd_max = tsd;
            if (tsd < res.tsd_min) res.tsd_min = tsd;
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

        // ---- band-parallel dispatch ----------------------------------------
        // The persistent team + dynamic-tile mechanism (parallelRange +
        // ChunkResult) lives in sim/threading.zig; the sim-specific worker
        // bodies stay here. Convention: `xxxRowsWorker` adapts `xxxRows`
        // (the banded loop body) to the dispatch signature.

        fn u2pRowsWorker(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            self.u2pRows(iy0, iy1) catch |e| {
                res.err = e;
            };
        }

        /// C: calc_u2p (finite.c:546) — per-cell inversion + floors, then
        /// fixup averaging and a boundary refresh.
        pub fn calcU2p(self: *Self) Error!void {
            self.timers.begin(.u2p);
            defer self.timers.end();
            const res = threading.parallelRange(Self, self, self.team, 0, self.nyi(), u2pRowsWorker);
            if (res.err) |e| return e;

            try self.cellFixup(.hd_fixup);
            if (comptime L.hasVar(.ee)) {
                try self.cellFixup(.rad_fixup);
            }
            try self.setBc(self.time, false);
        }

        /// The per-cell inversion body for iy ∈ [iy0, iy1) (all iz, all ix).
        fn u2pRows(self: *Self, iy0: i64, iy1: i64) Error!void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        var uu: [NV]f64 = undefined;
                        var pp: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);
                        self.p.load(ix, iy, iz, &pp);

                        self.setFlag(.entropy, ix, iy, iz, 0);
                        self.setFlag(.entropy2, ix, iy, iz, 0);

                        const geom = self.cache.fillGeometry(ix, iy, iz);

                        if (self.isCellCorrectedPolaraxis(iy)) {
                            // C: u2p_solver_Bonly (u2p.c:57) — invert only B
                            // (the rest is overwritten by do_correct); both
                            // floor checks skipped (u2p.c:80-92); corrected/
                            // fixups stay 0 so all flags read 0.
                            if (comptime L.hasVar(.b1)) {
                                const gdetu_inv = 1.0 / geom.gdet; // GDETIN == 1
                                pp[L.index(.b1)] = uu[L.index(.b1)] * gdetu_inv;
                                pp[L.index(.b2)] = uu[L.index(.b2)] * gdetu_inv;
                                pp[L.index(.b3)] = uu[L.index(.b3)] * gdetu_inv;
                            }
                            self.p.store(ix, iy, iz, &pp);
                            self.setFlag(.hd_fixup, ix, iy, iz, 0);
                            if (comptime L.hasVar(.ee)) {
                                self.setFlag(.rad_fixup, ix, iy, iz, 0);
                            }
                            continue;
                        }

                        const res = invert.u2pMhd(cfg, uu, &pp, &geom, self.opt.gam, self.opt.floors);
                        if (res.entropy_used) self.setFlag(.entropy, ix, iy, iz, 1);

                        // radiative closed-form inversion (u2p.c:386); runs
                        // regardless of the MHD outcome, on the same uu
                        var rad_fixup = false;
                        if (comptime L.hasVar(.ee)) {
                            const rr = invert_rad.u2pRad(cfg, uu, &pp, &geom, self.opt.rad);
                            rad_fixup = rr.corrected;
                        }

                        _ = try invert.checkFloorsMhd(cfg, &pp, &geom, self.opt.gam, self.opt.floors);
                        if (comptime L.hasVar(.ee)) {
                            _ = try invert_rad.checkFloorsRad(cfg, &pp, &geom, self.opt.rad);
                        }

                        self.p.store(ix, iy, iz, &pp);
                        self.setFlag(.hd_fixup, ix, iy, iz, if (res.fixup) 1 else 0);
                        if (comptime L.hasVar(.ee)) {
                            // C sets RADFIXUPFLAG to −1 (not 1) on request
                            self.setFlag(.rad_fixup, ix, iy, iz, if (rad_fixup) -1 else 0);
                        }
                    }
                }
            }
        }

        /// C: cell_fixup (finite.c:5012), types FIXUP_U2PMHD / FIXUP_U2PRAD.
        /// Averages flagged cells from their non-flagged in-domain neighbors.
        /// U2PMHD never averages rho or the magnetic field (iv != RHO &&
        /// iv < B1); U2PRAD averages only the radiative block (EE..FZ).
        /// True if any domain cell carries flag `which`. Early-out scan for
        /// cellFixup — reads one i32/cell (getFlag is inline) and returns on the
        /// first hit. Serial: the scan is ~NV× cheaper than the copies it may
        /// save, and short-circuits (P2 #3).
        fn anyFlagSet(self: *const Self, comptime which: Flag) bool {
            const nx = self.nxi();
            const ny = self.nyi();
            const nz = self.nzi();
            var iz: i64 = 0;
            while (iz < nz) : (iz += 1) {
                var iy: i64 = 0;
                while (iy < ny) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < nx) : (ix += 1) {
                        if (self.getFlag(which, ix, iy, iz) != 0) return true;
                    }
                }
            }
            return false;
        }

        pub fn cellFixup(self: *Self, comptime which: Flag) Error!void {
            const enabled = switch (which) {
                // C: DOFIXUPS && DOU2PMHDFIXUPS / DOU2PRADFIXUPS / DORADIMPFIXUPS
                .hd_fixup => self.opt.do_fixups,
                .rad_fixup => self.opt.do_fixups and self.opt.do_u2prad_fixups,
                .radimp_fixup => self.opt.do_fixups and self.opt.do_radimp_fixups,
                else => @compileError("cellFixup: bad type"),
            };
            if (!enabled) return;
            self.timers.begin(.fixup);
            defer self.timers.end();

            // Common case: no cell carries `which` this pass. Then fixupRows
            // writes nothing (every cell hits `getFlag == 0 → continue`), so the
            // four full-grid copies below (u↔u_bak, p↔p_bak — ~4×NV×cells×8 B,
            // ~350 MB/step at production size across the ~6 hd + radimp calls)
            // are pure churn. Scan the flag column first — one i32/cell, ~100×
            // less traffic than the copies — and skip the whole staging dance.
            // Bit-identical: with nothing flagged, u/p are unchanged either way
            // (P2 #3).
            if (!self.anyFlagSet(which)) return;

            threading.parallelCopy(self.team, self.u_bak.data, self.u.data);
            threading.parallelCopy(self.team, self.p_bak.data, self.p.data);

            // the averaging loop is parallel-safe as-is: flags are frozen
            // during the pass, reads touch only non-flagged neighbours of
            // the (frozen) p, writes only the flagged cells' _bak slots
            const res = threading.parallelRange(Self, self, self.team, 0, self.nyi(), fixupRowsWorker(which));
            if (res.err) |e| return e;

            threading.parallelCopy(self.team, self.u.data, self.u_bak.data);
            threading.parallelCopy(self.team, self.p.data, self.p_bak.data);
        }

        fn fixupRowsWorker(comptime which: Flag) fn (*Self, i64, i64, *threading.ChunkResult) void {
            return struct {
                fn w(s: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
                    s.fixupRows(which, iy0, iy1) catch |e| {
                        res.err = e;
                    };
                }
            }.w;
        }

        fn fixupRows(self: *Self, comptime which: Flag, iy0: i64, iy1: i64) Error!void {
            const b1_bound: usize = comptime if (L.hasVar(.b1)) L.index(.b1) else L.count;
            const nx = self.nxi();
            const ny = self.nyi();
            const nz = self.nzi();

            var iz: i64 = 0;
            while (iz < nz) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < nx) : (ix += 1) {
                        // C: "do not correct if overwritten later on"
                        if (self.isCellCorrectedPolaraxis(iy)) continue;
                        if (self.getFlag(which, ix, iy, iz) == 0) continue;

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
                            if (self.getFlag(which, nb[0], nb[1], nb[2]) != 0) continue;
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
                            // `which` is comptime — only the taken arm is analyzed
                            const fixthis = switch (which) {
                                .hd_fixup => iv != L.index(.rho) and iv < b1_bound,
                                .rad_fixup => iv >= L.index(.ee) and iv <= L.index(.fz),
                                // FIXUP_RADIMP: both fluids, never rho/B
                                .radimp_fixup => iv != L.index(.rho) and
                                    (iv < b1_bound or (iv >= L.index(.ee) and iv <= L.index(.fz))),
                                else => unreachable,
                            };
                            if (fixthis) {
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

            const kr_blk = self.cache.krBlock(ix, iy, iz);
            const rows = [4]usize{ L.index(.uu), L.index(.vx), L.index(.vy), L.index(.vz) };
            if (comptime L.hasVar(.ee)) {
                const rij_up = try radiation.calcRij(cfg, pp, &geom);
                const rij = relele.indices2221(rij_up, &geom.gg);
                const rrows = [4]usize{ L.index(.ee), L.index(.fx), L.index(.fy), L.index(.fz) };
                for (0..4) |k| {
                    for (0..4) |l| {
                        for (0..4) |nu| {
                            ss[rows[nu]] += gdetu * t[k][l] * kr_blk[l * 16 + nu * 4 + k];
                            ss[rrows[nu]] += gdetu * rij[k][l] * kr_blk[l * 16 + nu * 4 + k];
                        }
                    }
                }
            } else {
                for (0..4) |k| {
                    for (0..4) |l| {
                        for (0..4) |nu| {
                            ss[rows[nu]] += gdetu * t[k][l] * kr_blk[l * 16 + nu * 4 + k];
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
        /// Face-averaged viscous R^i_j at the face (fx,fy,fz) in dimension
        /// `dim` (between that cell and its dim−1 neighbour), C:
        /// f_flux_prime_rad_total's ifacedim>−1 branch (rad.c:3782-3786).
        fn rijviscFace(self: *const Self, dim: usize, fx: i64, fy: i64, fz: i64) [4][4]f64 {
            var a: [16]f64 = undefined;
            var b: [16]f64 = undefined;
            self.rijvisc.load(fx, fy, fz, &a);
            var c = [3]i64{ fx, fy, fz };
            c[dim] -= 1;
            self.rijvisc.load(c[0], c[1], c[2], &b);
            var out: [4][4]f64 = undefined;
            for (0..4) |i| {
                for (0..4) |j| out[i][j] = 0.5 * (a[i * 4 + j] + b[i * 4 + j]);
            }
            return out;
        }

        /// The two cross directions of a sweep/face pass in dimension `dim`.
        /// Band-parallelism runs over cross[0] — the largest transverse axis
        /// in 2D (y for the x-sweep, x for the y-sweep).
        fn crossDims(comptime dim: usize) [2]usize {
            return switch (dim) {
                0 => .{ 1, 2 },
                1 => .{ 0, 2 },
                2 => .{ 0, 1 },
                else => unreachable,
            };
        }

        fn sweep(self: *Self, comptime dim: usize) Error!void {
            const n = self.nDim(dim);
            if (n <= 1) return;
            self.timers.begin(.sweep);
            defer self.timers.end();
            const cross = comptime crossDims(dim);
            const res = threading.parallelRange(Self, self, self.team, self.crossLo(cross[0]), self.crossHi(cross[0]), sweepBandWorker(dim));
            if (res.err) |e| return e;
        }

        fn sweepBandWorker(comptime dim: usize) fn (*Self, i64, i64, *threading.ChunkResult) void {
            return struct {
                fn w(s: *Self, c0_lo: i64, c0_hi: i64, res: *threading.ChunkResult) void {
                    s.sweepBand(dim, c0_lo, c0_hi) catch |e| {
                        res.err = e;
                    };
                }
            }.w;
        }

        /// One direction of the interpolation sweep for cross[0] ∈
        /// [c0_lo, c0_hi): every (c0, c1) line's faces are written only by its
        /// own band, so the banding is bit-identical to serial. Loop order keeps
        /// the contiguous x index innermost — for dim==0 that is the sweep
        /// direction itself; for dim!=0 x is cross[0] (the band range this
        /// worker owns), so we iterate it innermost with the sweep direction in
        /// the middle, turning the five-point stencil into contiguous x streams
        /// instead of iy/iz strides (out of L2 on the 2D grid, a DRAM/TLB cliff
        /// in 3D). Each (i,c0,c1) writes only its own face slots, so reordering
        /// the iterations is bit-identical (P2 #5).
        fn sweepBand(self: *Self, comptime dim: usize, c0_lo: i64, c0_hi: i64) Error!void {
            const n = self.nDim(dim);
            const cross = comptime crossDims(dim);
            const c1_lo = self.crossLo(cross[1]);
            const c1_hi = self.crossHi(cross[1]);

            if (comptime dim == 0) {
                var c0 = c0_lo;
                while (c0 < c0_hi) : (c0 += 1) {
                    var c1 = c1_lo;
                    while (c1 < c1_hi) : (c1 += 1) {
                        var i: i64 = -1;
                        while (i < n + 1) : (i += 1) {
                            try self.sweepFace(dim, i, c0, c1, self.dx5At(dim, i));
                        }
                    }
                }
            } else {
                var c1 = c1_lo;
                while (c1 < c1_hi) : (c1 += 1) {
                    var i: i64 = -1;
                    while (i < n + 1) : (i += 1) {
                        const dx5 = self.dx5At(dim, i); // invariant over the inner (x) loop — hoisted (#6)
                        var c0 = c0_lo;
                        while (c0 < c0_hi) : (c0 += 1) {
                            try self.sweepFace(dim, i, c0, c1, dx5);
                        }
                    }
                }
            }
        }

        /// The ±2 sweep-direction cell sizes at slot i (C: get_size_x). Function
        /// of (i,dim) only, so it lifts out of the cross loops.
        inline fn dx5At(self: *const Self, comptime dim: usize, i: i64) [5]f64 {
            return .{
                self.grid.cellSize(i - 2, dim),
                self.grid.cellSize(i - 1, dim),
                self.grid.cellSize(i, dim),
                self.grid.cellSize(i + 1, dim),
                self.grid.cellSize(i + 2, dim),
            };
        }

        /// Reconstruct + one-sided fluxes for the face pair bracketing the cell
        /// at (i along dim, c0=cross[0], c1=cross[1]); dx5 is the sweep-direction
        /// stencil spacing. `inline` so the body stays fused into sweepBand's
        /// loops (Debug too). All writes land in this cell's own face slots.
        inline fn sweepFace(self: *Self, comptime dim: usize, i: i64, c0: i64, c1: i64, dx5: [5]f64) Error!void {
            const n = self.nDim(dim);
            const cross = comptime crossDims(dim);
            var cell = [3]i64{ 0, 0, 0 };
            cell[dim] = i;
            cell[cross[0]] = c0;
            cell[cross[1]] = c1;

            const dol = i >= 0;
            const dor = i < n;

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
                // radiative viscosity: face i is between cells i−1 and i
                if (self.opt.radviscosity and comptime L.hasVar(.ee)) {
                    const rv = self.rijviscFace(dim, cell[0], cell[1], cell[2]);
                    try radvisc_mod.addRadViscFlux(Self, self, &ffl, &pl, &geom, dim, &rv);
                }
            }
            if (dor) {
                var cp = cell;
                cp[dim] = i + 1;
                const geom = self.cache.fillGeometryFace(cp[0], cp[1], cp[2], dim);
                _ = try invert.checkFloorsMhd(cfg, &pr, &geom, self.opt.gam, self.opt.floors);
                ffr = try flux_mod.fFluxPrime(cfg, pr, dim, &geom, self.opt.gam);
                // radiative viscosity: face i+1 is between cells i and i+1
                if (self.opt.radviscosity and comptime L.hasVar(.ee)) {
                    const rv = self.rijviscFace(dim, cp[0], cp[1], cp[2]);
                    try radvisc_mod.addRadViscFlux(Self, self, &ffr, &pr, &geom, dim, &rv);
                }
            }

            // pbR[dim] at face i ← pl; pbL[dim] at face i+1 ← pr
            var cp = cell;
            self.pb_r[dim].store(cp[0], cp[1], cp[2], &pl);
            if (dol) self.fl_r[dim].store(cp[0], cp[1], cp[2], &ffl);
            cp[dim] = i + 1;
            self.pb_l[dim].store(cp[0], cp[1], cp[2], &pr);
            if (dor) self.fl_l[dim].store(cp[0], cp[1], cp[2], &ffr);
        }

        /// C: f_calc_fluxes_at_faces (finite.c:1461) — for every face,
        /// combine the one-sided fluxes with LAXF or HLL using the saved
        /// cell wavespeeds. Hydro and radiation rows use separate speeds.
        /// Band-parallel over cross[0] per dimension, like the sweep.
        fn fluxesAtFaces(self: *Self) Error!void {
            self.timers.begin(.fluxes);
            defer self.timers.end();
            for (0..3) |d| threading.parallelZero(self.team, self.flb[d].data);

            inline for (0..3) |dim| {
                if (self.nDim(dim) > 1) {
                    const cross = comptime crossDims(dim);
                    const res = threading.parallelRange(Self, self, self.team, self.crossLo(cross[0]), self.crossHi(cross[0]), fluxesBandWorker(dim));
                    if (res.err) |e| return e;
                }
            }
        }

        fn fluxesBandWorker(comptime dim: usize) fn (*Self, i64, i64, *threading.ChunkResult) void {
            return struct {
                fn w(s: *Self, c0_lo: i64, c0_hi: i64, res: *threading.ChunkResult) void {
                    s.fluxesBand(dim, c0_lo, c0_hi) catch |e| {
                        res.err = e;
                    };
                }
            }.w;
        }

        /// Band-parallel flux combination over cross[0] ∈ [c0_lo, c0_hi). Same
        /// x-innermost loop order as sweepBand (P2 #5): dim==0 has the face index
        /// i=x innermost already; for dim!=0 the band range x=cross[0] iterates
        /// innermost with the face index in the middle, so the per-face reads and
        /// the flb store stream x. Each face writes only its own flb[dim] slot →
        /// order-independent, bit-identical.
        fn fluxesBand(self: *Self, comptime dim: usize, c0_lo: i64, c0_hi: i64) Error!void {
            const n = self.nDim(dim);
            const cross = comptime crossDims(dim);
            const c1_lo = self.crossLo(cross[1]);
            const c1_hi = self.crossHi(cross[1]);

            if (comptime dim == 0) {
                var c0 = c0_lo;
                while (c0 < c0_hi) : (c0 += 1) {
                    var c1 = c1_lo;
                    while (c1 < c1_hi) : (c1 += 1) {
                        var i: i64 = 0;
                        while (i <= n) : (i += 1) try self.fluxFace(dim, i, c0, c1);
                    }
                }
            } else {
                var c1 = c1_lo;
                while (c1 < c1_hi) : (c1 += 1) {
                    var i: i64 = 0;
                    while (i <= n) : (i += 1) {
                        var c0 = c0_lo;
                        while (c0 < c0_hi) : (c0 += 1) try self.fluxFace(dim, i, c0, c1);
                    }
                }
            }
        }

        /// Combine the one-sided fluxes at the single face (i along dim,
        /// c0=cross[0], c1=cross[1]) into flb[dim] via LAXF/HLL. `inline` to stay
        /// fused into fluxesBand's loops; writes only this face's own flb slot.
        inline fn fluxFace(self: *Self, comptime dim: usize, i: i64, c0: i64, c1: i64) Error!void {
            const cross = comptime crossDims(dim);
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

            // hydro and radiation are two separate systems,
            // each combined with its own speeds (finite.c:1549)
            const ag = @max(ap1, am1);
            const al = @min(ap1l, am1l);
            const ar = @max(ap1r, am1r);

            var agr: f64 = 0;
            var alr: f64 = 0;
            var arr: f64 = 0;
            if (comptime L.hasVar(.ee)) {
                const rp1l = self.scGet(arad_l[dim], cf[0], cf[1], cf[2]);
                const rp1r = self.scGet(arad_r[dim], cf[0], cf[1], cf[2]);
                const rp1 = self.scGet(arad_m[dim], cf[0], cf[1], cf[2]);
                const rm1l = self.scGet(arad_l[dim], cm[0], cm[1], cm[2]);
                const rm1r = self.scGet(arad_r[dim], cm[0], cm[1], cm[2]);
                const rm1 = self.scGet(arad_m[dim], cm[0], cm[1], cm[2]);
                agr = @max(rp1, rm1);
                alr = @min(rp1l, rm1l);
                arr = @max(rp1r, rm1r);
            }

            // load the two one-sided flux vectors once, combine into a stack
            // buffer, store the face once — vs per-iv get/set offset recompute
            // (P2 #4). uLl/uRl are already whole-cell arrays.
            var fll: [NV]f64 = undefined;
            var flr: [NV]f64 = undefined;
            self.fl_l[dim].load(cf[0], cf[1], cf[2], &fll);
            self.fl_r[dim].load(cf[0], cf[1], cf[2], &flr);
            var fstar: [NV]f64 = undefined;
            for (0..NV) |iv| {
                // C: i < NVMHD → hydro block, else radiation
                const is_rad = if (comptime L.hasVar(.ee)) iv >= L.index(.ee) else false;
                const ag_sel = if (is_rad) agr else ag;
                const al_sel = if (is_rad) alr else al;
                const ar_sel = if (is_rad) arr else ar;
                fstar[iv] = switch (comptime cfg.flux) {
                    .laxf => laxf_mod.laxf(fll[iv], flr[iv], uLl[iv], uRl[iv], ag_sel),
                    .hll => laxf_mod.hll(fll[iv], flr[iv], uLl[iv], uRl[iv], al_sel, ar_sel),
                };
            }
            self.flb[dim].store(cf[0], cf[1], cf[2], &fstar);
        }

        /// C: op_explicit (finite.c:633).
        pub fn opExplicit(self: *Self, t: f64, dtin: f64) Error!void {
            _ = t;
            // (upreexplicit/ppreexplicit copies skipped — see header)
            // calc_avgs_throughout: CALCHRONTHEGO only (M12)

            try self.calcWavespeeds();

            inline for (0..3) |dim| try self.sweep(dim);

            try self.fluxesAtFaces();

            if (comptime cfg.has(.mhd)) {
                self.timers.begin(.fluxct);
                defer self.timers.end();
                ct.fluxCt(Self, self);
            }

            // conserved update: du from flux divergence + source terms
            {
                self.timers.begin(.update);
                defer self.timers.end();
                var ctx = DtCtx{ .sim = self, .dt = dtin };
                const res = threading.parallelRange(DtCtx, &ctx, self.team, 0, self.nyi(), updateRowsWorker);
                if (res.err) |e| return e;
            }

            // (EXPLICIT_LAB_RAD_SOURCE not used — PUFFY couples implicitly)

            try self.calcU2p();
            // (MIXENTROPIESPROPERLY off)
        }

        fn updateRowsWorker(ctx: *DtCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            ctx.sim.updateRows(ctx.dt, iy0, iy1) catch |e| {
                res.err = e;
            };
        }

        /// The conserved update (flux divergence + metric source) for
        /// iy ∈ [iy0, iy1) — writes only its own cells' u.
        fn updateRows(self: *Self, dtin: f64, iy0: i64, iy1: i64) Error!void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        const ms = try self.metricSource(ix, iy, iz);
                        const dx = self.grid.cellSize(ix, 0);
                        const dy = self.grid.cellSize(iy, 1);
                        const dz = self.grid.cellSize(iz, 2);
                        const dt = dtin;

                        // Load each accessed face/cell vector once into a stack
                        // buffer (single offset per cell) and index the buffers
                        // per iv — vs re-deriving the offset NV× per FaceStore.get
                        // (P2 #4). Per-component arithmetic unchanged → bit-identical.
                        var flxl: [NV]f64 = undefined;
                        var flxr: [NV]f64 = undefined;
                        var flyl: [NV]f64 = undefined;
                        var flyr: [NV]f64 = undefined;
                        var flzl: [NV]f64 = undefined;
                        var flzr: [NV]f64 = undefined;
                        self.flb[0].load(ix, iy, iz, &flxl);
                        self.flb[0].load(ix + 1, iy, iz, &flxr);
                        self.flb[1].load(ix, iy, iz, &flyl);
                        self.flb[1].load(ix, iy + 1, iz, &flyr);
                        self.flb[2].load(ix, iy, iz, &flzl);
                        self.flb[2].load(ix, iy, iz + 1, &flzr);
                        var uu: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);

                        for (0..NV) |iv| {
                            const du = -(flxr[iv] - flxl[iv]) * dt / dx - (flyr[iv] - flyl[iv]) * dt / dy - (flzr[iv] - flzl[iv]) * dt / dz;
                            uu[iv] = uu[iv] + du + ms[iv] * dt;
                        }
                        self.u.store(ix, iy, iz, &uu);
                    }
                }
            }
        }

        // ---- stage arithmetic (exact C expression shapes) ------------------------

        fn copyFull(self: *Self, dst: *FieldT, src: *const FieldT) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            threading.parallelCopy(self.team, dst.data, src.data);
        }

        /// Worker context for the two stage-arithmetic shapes: deriv uses
        /// (dst, a, b, f1, f2); combine additionally uses c.
        const StageCtx = struct {
            sim: *Self,
            dst: *FieldT,
            a_f: *const FieldT,
            b_f: *const FieldT,
            c_f: *const FieldT = undefined,
            f1: f64,
            f2: f64,
        };

        /// dst = (1/Δ)·a + (−1/Δ)·b over the domain (problem.c:181/230/…).
        /// Band-parallel over iy — pure per-cell assignments.
        fn stageDeriv(self: *Self, dst: *FieldT, a_f: *const FieldT, b_f: *const FieldT, delta: f64) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            var ctx = StageCtx{ .sim = self, .dst = dst, .a_f = a_f, .b_f = b_f, .f1 = 1.0 / delta, .f2 = -1.0 / delta };
            _ = threading.parallelRange(StageCtx, &ctx, self.team, 0, self.nyi(), stageDerivWorker);
        }

        fn stageDerivWorker(ctx: *StageCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            const self = ctx.sim;
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        // one offset per cell, index the stack buffers per iv (P2 #4)
                        var a_v: [NV]f64 = undefined;
                        var b_v: [NV]f64 = undefined;
                        ctx.a_f.load(ix, iy, iz, &a_v);
                        ctx.b_f.load(ix, iy, iz, &b_v);
                        var d_v: [NV]f64 = undefined;
                        for (0..NV) |iv| d_v[iv] = ctx.f1 * a_v[iv] + ctx.f2 * b_v[iv];
                        ctx.dst.store(ix, iy, iz, &d_v);
                    }
                }
            }
        }

        /// dst = a + f1·b + f2·c over the domain (problem.c:243/357).
        fn stageCombine(self: *Self, dst: *FieldT, a_f: *const FieldT, f1: f64, b_f: *const FieldT, f2: f64, c_f: *const FieldT) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            var ctx = StageCtx{ .sim = self, .dst = dst, .a_f = a_f, .b_f = b_f, .c_f = c_f, .f1 = f1, .f2 = f2 };
            _ = threading.parallelRange(StageCtx, &ctx, self.team, 0, self.nyi(), stageCombineWorker);
        }

        fn stageCombineWorker(ctx: *StageCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            const self = ctx.sim;
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        // one offset per cell, index the stack buffers per iv (P2 #4)
                        var a_v: [NV]f64 = undefined;
                        var b_v: [NV]f64 = undefined;
                        var c_v: [NV]f64 = undefined;
                        ctx.a_f.load(ix, iy, iz, &a_v);
                        ctx.b_f.load(ix, iy, iz, &b_v);
                        ctx.c_f.load(ix, iy, iz, &c_v);
                        var d_v: [NV]f64 = undefined;
                        for (0..NV) |iv| d_v[iv] = a_v[iv] + ctx.f1 * b_v[iv] + ctx.f2 * c_v[iv];
                        ctx.dst.store(ix, iy, iz, &d_v);
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
        /// MHD conserveds. Band-parallel over iy (cell-local).
        pub fn updateEntropy(self: *Self) Error!void {
            self.timers.begin(.entropy);
            defer self.timers.end();
            const res = threading.parallelRange(Self, self, self.team, 0, self.nyi(), entropyRowsWorker);
            if (res.err) |e| return e;
        }

        fn entropyRowsWorker(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            self.entropyRows(iy0, iy1) catch |e| {
                res.err = e;
            };
        }

        fn entropyRows(self: *Self, iy0: i64, iy1: i64) Error!void {
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
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

        /// C: op_implicit (finite.c:1400) — the implicit radiative source
        /// operator: per-cell solve_implicit_lab, RADIMPFIXUPFLAG, then
        /// cell_fixup(FIXUP_RADIMP). Structural no-op without radiation or
        /// when opt.opac is null (≡ SKIPRADSOURCE).
        pub fn opImplicit(self: *Self, t: f64, dtin: f64) Error!void {
            _ = t;
            if (comptime !L.hasVar(.ee)) return;
            _ = self.opt.opac orelse return;

            self.timers.begin(.implicit);
            defer self.timers.end();
            var ctx = DtCtx{ .sim = self, .dt = dtin };
            const res = threading.parallelRange(DtCtx, &ctx, self.team, 0, self.nyi(), implicitRowsWorker);
            self.n_radimp_failures += res.n_fail;
            self.n_radimp_iters += res.n_iters;
            self.n_radimp_solves += res.n_solves;
            if (res.err) |e| return e;

            try self.cellFixup(.radimp_fixup);
        }

        /// Worker context for the passes parameterized by the sub-step dt
        /// (op_implicit, the conserved update).
        const DtCtx = struct { sim: *Self, dt: f64 };

        fn implicitRowsWorker(ctx: *DtCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            // errors here would be relele failures inside the solver, but
            // solve_implicit_lab returns a status (never throws), so this
            // worker cannot error — it only tallies counters into `res`.
            const self = ctx.sim;
            const opac = &(self.opt.opac.?);
            const ImplT = implicit.Solver(cfg);

            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        // C: finite.c:1427 — polar-corrected cells skip the
                        // implicit entirely (is_cell_active ≡ 1 and PUFFY
                        // defines no SKIPIMPLICIT_* hooks)
                        if (self.isCellCorrectedPolaraxis(iy)) continue;

                        var uu: [NV]f64 = undefined;
                        var pp: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);
                        self.p.load(ix, iy, iz, &pp);

                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const rr = ImplT.solveImplicitLab(&uu, &pp, &geom, ctx.dt, self.opt.gam, self.opt.rad, opac, &self.opt.implicit);

                        if (rr.ok) {
                            self.u.store(ix, iy, iz, &uu);
                            self.p.store(ix, iy, iz, &pp);
                            self.setFlag(.radimp_fixup, ix, iy, iz, 0);
                            res.n_iters += rr.iters;
                            res.n_solves += 1;
                        } else {
                            // C: unchanged u/p, flag for fixups
                            self.setFlag(.radimp_fixup, ix, iy, iz, -1);
                            res.n_fail += 1;
                        }
                    }
                }
            }
        }

        /// C: do_correct (finite.c:594) — CORRECT_POLARAXIS only; the 3D /
        /// smoothing / NS-surface variants are not PUFFY machinery.
        pub fn doCorrect(self: *Self) Error!void {
            if (self.opt.correct_polaraxis) {
                self.timers.begin(.correct);
                defer self.timers.end();
                try self.correctPolaraxis();
            }
        }

        /// C: correct_polaraxis (finite.c:5525) — the NCCORRECTPOLAR
        /// most-polar rows are not evolved but overwritten per (ix,iz)
        /// column from row nc: scalars and in-row velocities copied, the
        /// θ-components scaled by |θ−θ_axis|/|θ_src−θ_axis| (internal x2),
        /// then p2u at the target geometry rewrites all conserveds. B is
        /// untouched: PUFFY does not define CORRECTMAGNFIELD. C only acts
        /// on spherical-like MYCOORDS. Band-parallel over ix columns (each
        /// column reads its own row-nc source and writes its own polar rows).
        fn correctPolaraxis(self: *Self) Error!void {
            if (comptime cfg.coords == .mink) return;
            const res = threading.parallelRange(Self, self, self.team, 0, self.nxi(), polarColsWorker);
            if (res.err) |e| return e;
        }

        fn polarColsWorker(self: *Self, ix0: i64, ix1: i64, res: *threading.ChunkResult) void {
            self.polarCols(ix0, ix1) catch |e| {
                res.err = e;
            };
        }

        fn polarCols(self: *Self, ix0: i64, ix1: i64) Error!void {
            const g = &self.grid;
            const nc = self.opt.nccorrectpolar;
            const ny = self.nyi();

            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var ix: i64 = ix0;
                while (ix < ix1) : (ix += 1) {
                    // upper axis
                    {
                        const thaxis = g.yl(0);
                        var ic: i64 = 0;
                        while (ic < nc) : (ic += 1) {
                            const iy = ic;
                            const iysrc = nc;
                            const th = g.yc(iy);
                            const thsrc = g.yc(iysrc);
                            const fac = @abs((th - thaxis) / (thsrc - thaxis));
                            try self.polarOverwriteCell(ix, iy, iz, iysrc, fac);
                        }
                    }
                    // lower axis (no HALFTHETA)
                    {
                        const thaxis = g.yl(ny);
                        var ic: i64 = 0;
                        while (ic < nc) : (ic += 1) {
                            const iy = ny - 1 - ic;
                            const iysrc = ny - 1 - nc;
                            const th = g.yc(iy);
                            const thsrc = g.yc(iysrc);
                            const fac = @abs((th - thaxis) / (thsrc - thaxis));
                            try self.polarOverwriteCell(ix, iy, iz, iysrc, fac);
                        }
                    }
                }
            }
        }

        fn polarOverwriteCell(self: *Self, ix: i64, iy: i64, iz: i64, iysrc: i64, fac: f64) Error!void {
            var pp: [NV]f64 = undefined;
            var pps: [NV]f64 = undefined;
            self.p.load(ix, iy, iz, &pp);
            self.p.load(ix, iysrc, iz, &pps);

            pp[L.index(.rho)] = pps[L.index(.rho)];
            pp[L.index(.uu)] = pps[L.index(.uu)];
            pp[L.index(.entr)] = pps[L.index(.entr)];
            pp[L.index(.vx)] = pps[L.index(.vx)];
            pp[L.index(.vz)] = pps[L.index(.vz)];
            pp[L.index(.vy)] = fac * pps[L.index(.vy)];
            if (comptime L.hasVar(.ee)) {
                pp[L.index(.ee)] = pps[L.index(.ee)];
                pp[L.index(.fx)] = pps[L.index(.fx)];
                pp[L.index(.fz)] = pps[L.index(.fz)];
                pp[L.index(.fy)] = fac * pps[L.index(.fy)];
            }

            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
            self.p.store(ix, iy, iz, &pp);
            self.u.store(ix, iy, iz, &uu);
        }

        // ---- one full RK2IMEX step (problem.c:141-402) ----------------------------

        pub fn step(self: *Self, forced_dt: ?f64) Error!void {
            comptime std.debug.assert(cfg.timestepping == .rk2imex);

            self.timers.begin(.step);
            defer self.timers.end();

            const t = self.t;
            self.time = t; // C: global_time = t

            // The CFL denominator must have been seeded (initTimestepGuess, part
            // of C's solve preamble) or be forced — otherwise tstepdenmax is 0
            // and own_dt is +inf → NaNs on the first step, diagnosed only far
            // downstream. Turn that ordering bug into an immediate failure.
            std.debug.assert(forced_dt != null or self.tstepdenmax > 0);
            self.own_dt = self.cflDt();
            const dt = forced_dt orelse self.own_dt;
            self.dt = dt;

            // reset wavespeed accumulators for this step
            self.tstepdenmax = -1;
            self.tstepdenmin = big;

            // C: calc_Rij_visc_total (problem.c:127) — once per step, with
            // global_dt = this step's dt, feeding the RADVISCNUDAMP cap.
            if (self.opt.radviscosity and comptime L.hasVar(.ee)) {
                self.timers.begin(.radvisc);
                defer self.timers.end();
                try radvisc_mod.calcRijViscTotal(Self, self, dt);
            }

            const gam_imex = 1.0 - 1.0 / @sqrt(2.0);

            // my_finger: PR_FINGER not defined for any target problem
            self.saveTimesteps();
            // set_gammagas: CONSISTENTGAMMA off

            // ---- 1st implicit ----
            self.copyFull(&self.ut0, &self.u);
            self.copyFull(&self.ptm1, &self.p);
            try self.opImplicit(t, dt * gam_imex); // U(1)
            self.copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt1, &self.u, &self.ut0, dt * gam_imex);

            // ---- 1st explicit ----
            self.copyFull(&self.ut1, &self.u);
            try self.calcU2p();
            try self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(1))
            if (self.opt.dynamo and comptime L.hasVar(.b1)) {
                self.timers.begin(.dynamo);
                defer self.timers.end();
                try dynamo_mod.applyDynamo(Self, self, t, dt);
            }
            // op_intermediate (electrons) — no-op
            self.copyEntropyCount();
            self.stageDeriv(&self.dut1, &self.u, &self.ut1, dt);

            // ---- 1st together: u = ut0 + dt·dut1 + dt(1−2γ)·drt1 ----
            self.stageCombine(&self.u, &self.ut0, dt, &self.dut1, dt * (1.0 - 2.0 * gam_imex), &self.drt1);

            // ---- 2nd implicit ----
            self.copyFull(&self.uforget, &self.u);
            try self.calcU2p();
            self.copyFull(&self.ptm1, &self.p);
            try self.doCorrect();
            try self.opImplicit(t, gam_imex * dt); // U(2)
            self.copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt2, &self.u, &self.uforget, dt * gam_imex);

            // ---- 2nd explicit ----
            self.copyFull(&self.ut2, &self.u);
            try self.calcU2p();
            try self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(2))
            if (self.opt.dynamo and comptime L.hasVar(.b1)) {
                self.timers.begin(.dynamo);
                defer self.timers.end();
                try dynamo_mod.applyDynamo(Self, self, t, dt);
            }
            self.stageDeriv(&self.dut2, &self.u, &self.ut2, dt);

            // ---- explicit together: u = ut0 + dt/2·(dut1 + dut2) ----
            self.stageCombine(&self.u, &self.ut0, dt / 2.0, &self.dut1, dt / 2.0, &self.dut2);
            // ---- implicit together: u += dt/2·(drt1 + drt2) ----
            self.stageCombine(&self.u, &self.u, dt / 2.0, &self.drt1, dt / 2.0, &self.drt2);

            // ---- final inversion & bookkeeping ----
            try self.calcU2p();
            try self.doCorrect();
            try self.setBc(t, false);

            self.t = t + dt;
            try self.updateEntropy();
            self.nstep += 1;
        }
    };
}

test {
    _ = FaceStore(5);
    _ = timers_mod;
}
