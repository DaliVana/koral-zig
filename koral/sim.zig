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

const Grid = grid_mod.Grid;
const Geometry = geometry.Geometry;

pub const Error = relele.Error || error{OutOfMemory};

const big: f64 = 1.0e50; // C: BIG (ko.h) — only compared against, value moot

// Storage & bookkeeping (face buffers, integer-flag / scalar-slot enums) live
// in sim/storage.zig; the boundary-condition enums + logic in sim/bc.zig.
// Re-exported here so external code keeps referencing them as sim.BcFace /
// sim.Flag / sim.FaceStore.
pub const BcKind = bc.BcKind;
pub const BcFace = bc.BcFace;
pub const FaceStore = storage.FaceStore;
pub const Flag = storage.Flag;
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
            /// Worker threads for the per-cell inversions (u2p / implicit).
            /// 1 ≡ the serial path (bit-identical; the golden tests run here).
            nthreads: usize = 1,
        };

        allocator: std.mem.Allocator,
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
        /// implicit-solver diagnostics (C: global_int_slot counters)
        n_radimp_failures: u64 = 0,
        n_radimp_iters: u64 = 0,
        /// transient: the dt passed to the current op_implicit parallel pass
        /// (the per-cell inversions are independent, so this is set once per
        /// pass and read by every worker row).
        impl_dt: f64 = 0,

        pub fn init(allocator: std.mem.Allocator, g: Grid, opt: Options) !Self {
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

            self.cache = try precompute.MetricCache.init(allocator, g, .{
                .coords = opt.coords,
                .out_coords = opt.coords,
                .mp = opt.mp,
            });

            inline for (.{
                "p",           "u",    "ut0",   "ut1",     "ut2", "dut1", "dut2",
                "drt1",        "drt2", "uforget", "ptm1",
                "ppostimplicit", "u_bak", "p_bak",
            }) |name| {
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
            inline for (.{
                "p",           "u",    "ut0",   "ut1",     "ut2", "dut1", "dut2",
                "drt1",        "drt2", "uforget", "ptm1",
                "ppostimplicit", "u_bak", "p_bak",
            }) |name| {
                @field(self, name).deinit();
            }
            for (0..3) |d| {
                self.pb_l[d].deinit(allocator);
                self.pb_r[d].deinit(allocator);
                self.fl_l[d].deinit(allocator);
                self.fl_r[d].deinit(allocator);
                self.flb[d].deinit(allocator);
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

        // ---- boundary conditions --------------------------------------------

        /// C: set_bc (finite.c:2805). The implementation (ghost fill + the 2D
        /// corner filling, and the MPI-growth seam) lives in sim/bc.zig; this
        /// is the stable method entry point that problems/tests call.
        pub fn setBc(self: *Self, t: f64, ifinit: bool) Error!void {
            return bc.setBc(Self, self, t, ifinit);
        }


        // ---- wavespeeds & timestep -------------------------------------------

        /// C: calc_wavespeeds (finite.c:356) over domain + 1 ghost layer.
        pub fn calcWavespeeds(self: *Self) Error!void {
            const active_dims = [3]bool{ self.grid.nx > 1, self.grid.ny > 1, self.grid.nz > 1 };
            const lx: i64 = if (active_dims[0]) 1 else 0;
            const ly: i64 = if (active_dims[1]) 1 else 0;
            const lz: i64 = if (active_dims[2]) 1 else 0;

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
                        const aaa = try wavespeeds.gasWavespeedsLr(cfg, pp, &geom, self.opt.gam, active_dims);

                        var rad0: [6]f64 = @splat(0);
                        var radl: [6]f64 = @splat(0);
                        if (comptime L.hasVar(.ee)) {
                            // optical depth over the cell (physics.c:549-558);
                            // ix could be a face index in C, hence the max of
                            // left/right sizes
                            var tautot: [3]f64 = @splat(0);
                            if (self.opt.opac) |*op| {
                                const chi = try radforce.calcChi(cfg, pp, &geom, self.opt.gam, op);
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
                        self.saveWavespeeds(ix, iy, iz, aaa, rad0, radl);
                    }
                }
            }
        }

        /// C: save_wavespeeds (finite.c:394) — store per-cell speeds and,
        /// for domain cells, the CFL denominator (feeding the next dt).
        /// rad0 (unlimited by τ) only enters the timestep; radl (τ-limited)
        /// is what the flux combination reads.
        fn saveWavespeeds(self: *Self, ix: i64, iy: i64, iz: i64, aaa: [6]f64, rad0: [6]f64, radl: [6]f64) void {
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

        // ---- optional row-parallel dispatch --------------------------------
        // The generic row-splitting mechanism (parallelRows + ChunkResult)
        // lives in sim/threading.zig; the sim-specific worker bodies stay here.

        fn u2pRowsWorker(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            self.u2pRows(iy0, iy1) catch |e| {
                res.err = e;
            };
        }

        /// C: calc_u2p (finite.c:546) — per-cell inversion + floors, then
        /// fixup averaging and a boundary refresh. The inversion sweep is
        /// row-parallel; the neighbour-averaging fixup and BC refresh stay
        /// serial (they read across rows).
        pub fn calcU2p(self: *Self) Error!void {
            try threading.parallelRows(Self, self, u2pRowsWorker);

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
        pub fn cellFixup(self: *Self, comptime which: Flag) Error!void {
            const enabled = switch (which) {
                // C: DOFIXUPS && DOU2PMHDFIXUPS / DOU2PRADFIXUPS / DORADIMPFIXUPS
                .hd_fixup => self.opt.do_fixups,
                .rad_fixup => self.opt.do_fixups and self.opt.do_u2prad_fixups,
                .radimp_fixup => self.opt.do_fixups and self.opt.do_radimp_fixups,
                else => @compileError("cellFixup: bad type"),
            };
            if (!enabled) return;

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
                const rij_up = try radiation.calcRij(cfg, pp, &geom);
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
                            self.grid.cellSize(i - 2, dim),
                            self.grid.cellSize(i - 1, dim),
                            self.grid.cellSize(i, dim),
                            self.grid.cellSize(i + 1, dim),
                            self.grid.cellSize(i + 2, dim),
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

                            for (0..NV) |iv| {
                                // C: i < NVMHD → hydro block, else radiation
                                const is_rad = if (comptime L.hasVar(.ee)) iv >= L.index(.ee) else false;
                                const ag_sel = if (is_rad) agr else ag;
                                const al_sel = if (is_rad) alr else al;
                                const ar_sel = if (is_rad) arr else ar;
                                const fl = self.fl_l[dim].get(iv, cf[0], cf[1], cf[2]);
                                const fr = self.fl_r[dim].get(iv, cf[0], cf[1], cf[2]);
                                const fstar = switch (comptime cfg.flux) {
                                    .laxf => laxf_mod.laxf(fl, fr, uLl[iv], uRl[iv], ag_sel),
                                    .hll => laxf_mod.hll(fl, fr, uLl[iv], uRl[iv], al_sel, ar_sel),
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
                        const dx = self.grid.cellSize(ix, 0);
                        const dy = self.grid.cellSize(iy, 1);
                        const dz = self.grid.cellSize(iz, 2);
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

        /// C: op_implicit (finite.c:1400) — the implicit radiative source
        /// operator: per-cell solve_implicit_lab, RADIMPFIXUPFLAG, then
        /// cell_fixup(FIXUP_RADIMP). Structural no-op without radiation or
        /// when opt.opac is null (≡ SKIPRADSOURCE).
        pub fn opImplicit(self: *Self, t: f64, dtin: f64) Error!void {
            _ = t;
            if (comptime !L.hasVar(.ee)) return;
            _ = self.opt.opac orelse return;

            self.impl_dt = dtin; // read by every worker row (single pass)
            try threading.parallelRows(Self, self, implicitRowsWorker);

            try self.cellFixup(.radimp_fixup);
        }

        fn implicitRowsWorker(self: *Self, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            // errors here would be relele failures inside the solver, but
            // solve_implicit_lab returns a status (never throws), so this
            // worker cannot error — it only tallies counters into `res`.
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
                        const rr = ImplT.solveImplicitLab(&uu, &pp, &geom, self.impl_dt, self.opt.gam, self.opt.rad, opac, &self.opt.implicit);

                        if (rr.ok) {
                            self.u.store(ix, iy, iz, &uu);
                            self.p.store(ix, iy, iz, &pp);
                            self.setFlag(.radimp_fixup, ix, iy, iz, 0);
                            res.n_iters += rr.iters;
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
            if (self.opt.correct_polaraxis) try self.correctPolaraxis();
        }

        /// C: correct_polaraxis (finite.c:5525) — the NCCORRECTPOLAR
        /// most-polar rows are not evolved but overwritten per (ix,iz)
        /// column from row nc: scalars and in-row velocities copied, the
        /// θ-components scaled by |θ−θ_axis|/|θ_src−θ_axis| (internal x2),
        /// then p2u at the target geometry rewrites all conserveds. B is
        /// untouched: PUFFY does not define CORRECTMAGNFIELD. C only acts
        /// on spherical-like MYCOORDS.
        fn correctPolaraxis(self: *Self) Error!void {
            if (comptime cfg.coords == .mink) return;
            const g = &self.grid;
            const nc = self.opt.nccorrectpolar;
            const ny = self.nyi();

            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var ix: i64 = 0;
                while (ix < self.nxi()) : (ix += 1) {
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

            const t = self.t;
            self.time = t; // C: global_time = t

            self.own_dt = 1.0 / self.tstepdenmax;
            const dt = forced_dt orelse self.own_dt;
            self.dt = dt;

            // reset wavespeed accumulators for this step
            self.tstepdenmax = -1;
            self.tstepdenmin = big;

            // C: calc_Rij_visc_total (problem.c:127) — once per step, with
            // global_dt = this step's dt, feeding the RADVISCNUDAMP cap.
            if (self.opt.radviscosity and comptime L.hasVar(.ee)) {
                try radvisc_mod.calcRijViscTotal(Self, self, dt);
            }

            const gam_imex = 1.0 - 1.0 / @sqrt(2.0);

            // my_finger: PR_FINGER not defined for any target problem
            self.saveTimesteps();
            // set_gammagas: CONSISTENTGAMMA off

            // ---- 1st implicit ----
            copyFull(&self.ut0, &self.u);
            copyFull(&self.ptm1, &self.p);
            try self.opImplicit(t, dt * gam_imex); // U(1)
            copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt1, &self.u, &self.ut0, dt * gam_imex);

            // ---- 1st explicit ----
            copyFull(&self.ut1, &self.u);
            try self.calcU2p();
            try self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(1))
            if (self.opt.dynamo and comptime L.hasVar(.b1)) try dynamo_mod.applyDynamo(Self, self, t, dt);
            // op_intermediate (electrons) — no-op
            self.copyEntropyCount();
            self.stageDeriv(&self.dut1, &self.u, &self.ut1, dt);

            // ---- 1st together: u = ut0 + dt·dut1 + dt(1−2γ)·drt1 ----
            self.stageCombine(&self.u, &self.ut0, dt, &self.dut1, dt * (1.0 - 2.0 * gam_imex), &self.drt1);

            // ---- 2nd implicit ----
            copyFull(&self.uforget, &self.u);
            try self.calcU2p();
            copyFull(&self.ptm1, &self.p);
            try self.doCorrect();
            try self.opImplicit(t, gam_imex * dt); // U(2)
            copyFull(&self.ppostimplicit, &self.p);
            self.stageDeriv(&self.drt2, &self.u, &self.uforget, dt * gam_imex);

            // ---- 2nd explicit ----
            copyFull(&self.ut2, &self.u);
            try self.calcU2p();
            try self.doCorrect();
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(2))
            if (self.opt.dynamo and comptime L.hasVar(.b1)) try dynamo_mod.applyDynamo(Self, self, t, dt);
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
}
