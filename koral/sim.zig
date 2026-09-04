//! The evolution driver: `Sim(cfg)` owns all grid state and is the stable
//! entry point for every pass; the RK2IMEX step is `step()`. Transcribed
//! from KORAL's serial path:
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
//! Where the per-cell work lives (sim.zig redesign step 2, 2026-09-04):
//! every pass body is a `sim/*.zig` module generic over SimT, and the
//! methods here are thin operators that own the timers and hand the pass's
//! banded worker to the team:
//!
//!   sim/timestep.zig     wavespeeds, CFL denominator, save_timesteps
//!   sim/u2p.zig          the per-cell inversion sweep      (calcU2p)
//!   sim/fixup.zig        cell_fixup staging + averaging    (cellFixup)
//!   sim/explicit.zig     sweep / fluxes / metric source / update (opExplicit)
//!   sim/implicit_op.zig  the per-cell implicit solve       (opImplicit)
//!   sim/entropy.zig      copy_entropycount, update_entropy
//!   sim/stage.zig        the RK stage arithmetic
//!   sim/bc.zig · ct.zig · dynamo.zig · rijvisc.zig · polaraxis.zig ·
//!   storage.zig · timers.zig                               (unchanged)
//!
//! C-fidelity notes:
//!  * "wide" sweep bounds mirror MPI4CORNERS, which choices.h:790 force-
//!    defines whenever MAGNFIELD is on; even in serial builds. So MHD
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
//!  * upreexplicit/ppreexplicit copies (finite.c:644) are skipped; nothing
//!    reads them before M12 (radviscosity / entropy mixing).

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const grid_mod = @import("grid.zig");
const field_mod = @import("field.zig");
const geometry = @import("geometry.zig");
const metric = @import("metric/metric.zig");
const coco = @import("metric/coco.zig");
const precompute = @import("metric/precompute.zig");
const relele = @import("relele.zig");
const invert = @import("solve/invert.zig");
const invert_rad = @import("solve/invert_rad.zig");
const implicit = @import("solve/implicit.zig");
const radforce = @import("physics/radforce.zig");
const radvisc_mod = @import("physics/radvisc.zig");
const rijvisc_mod = @import("sim/rijvisc.zig");
const p2u_mod = @import("p2u.zig");
const ct = @import("sim/ct.zig");
const dynamo_mod = @import("sim/dynamo.zig");
const storage = @import("sim/storage.zig");
const bc = @import("sim/bc.zig");
const polaraxis = @import("sim/polaraxis.zig");
const threading = @import("threading.zig");
const timers_mod = @import("sim/timers.zig");
const timestep = @import("sim/timestep.zig");
const u2p_pass = @import("sim/u2p.zig");
const fixup_pass = @import("sim/fixup.zig");
const explicit_pass = @import("sim/explicit.zig");
const implicit_op = @import("sim/implicit_op.zig");
const entropy_pass = @import("sim/entropy.zig");
const stage = @import("sim/stage.zig");
const comm_mod = @import("comm/comm.zig");

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
pub const Pass = timers_mod.Pass;
/// Monotonic wall clock in ns (sim/timers.zig); re-exported so the run
/// driver can time steps / throttle the heartbeat with the same clock.
pub const nowNs = timers_mod.nowNs;
const n_flags = storage.n_flags;
const Scal = storage.Scal;
const n_scal = storage.n_scal;

pub fn Sim(comptime cfg: config.Config) type {
    comptime cfg.validate();
    const L = layout.VarLayout(cfg);
    const NV = L.count;
    // C: MPI4CORNERS is force-defined by MAGNFIELD (choices.h:790).
    const mhd_wide = cfg.has(.mhd);
    const recon_order: u8 = switch (cfg.reconstruction) {
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
        /// sim/ct.zig's scratch type when the layout has B, else void.
        pub const CtScratch = if (L.hasVar(.b1)) ct.Scratch else void;
        /// C: MPI4CORNERS (force-defined by MAGNFIELD): MHD sweeps reach ±1
        /// ghost row and fill 2D corners. Read by the sim/ passes.
        pub const wide = mhd_wide;
        /// Reconstruction order 0/1/2 from cfg.reconstruction.
        pub const base_order: u8 = recon_order;

        /// A problem's boundary-condition hook. Returns the ghost-cell
        /// primitives, or an Error; the fallible frame/velocity conversions a
        /// boundary-adjacent cell can reach (SpacelikeVelocity,
        /// VelocityConversionFailed) propagate instead of being swallowed by
        /// `catch unreachable` (panic in safe builds, UB in ReleaseFast). The
        /// sole caller setBcCell already returns Error!void.
        pub const SpecificBc = *const fn (
            ctx: ?*const anyopaque,
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
            do_radimp_fixups: bool = false, // C: DORADIMPFIXUPS (off in the validated build; on in koral_lite_puffy)
            /// C: REDUCEORDERATBH. Drop the reconstruction order by one for
            /// cells whose center is inside the BL horizon (PPM→linear there).
            reduceorderatbh: bool = false,
            /// koral_lite_puffy REDUCEORDERAFTERFIXUP (finite.c:26-40,
            /// 2026-08-11): a cell whose most recent u2p / rad / implicit
            /// pass demanded a fixup reconstructs one order lower; its
            /// primitives are synthetic neighbour averages, and a high-order
            /// stencil over them tends to make the same cell fail again. One
            /// diffusive sweep lets it relax, then full order resumes.
            reduceorderafterfixup: bool = false,
            /// C: DAMPRADWAVESPEEDNEARAXIS + …NCELLS. Within this many cells of
            /// each pole, force the radiative wavespeed to the undamped 1/3.
            /// 0 = off.
            dampradwavespeednearaxis: usize = 0,
            /// C: CORRECT_POLARAXIS (PUFFY define.h:107). Do_correct
            /// overwrites the most-polar rows and u2p/implicit/fixups skip
            /// them. Only meaningful on spherical-like coordinates.
            correct_polaraxis: bool = false,
            /// C: NCCORRECTPOLAR (PUFFY: 2).
            nccorrectpolar: i64 = 2,
            /// C: RADVISCOSITY==SHEARVISCOSITY (PUFFY define.h:96). The
            /// radiative shear-viscosity contribution to R^ij, computed once
            /// per step (calc_Rij_visc_total) and added at the faces.
            radviscosity: bool = false,
            radvisc: radvisc_mod.Params = .{},
            /// C: MIMICDYNAMO (PUFFY define.h:66). The mean-field dynamo run
            /// after each explicit sub-step (problem.c:210,326).
            dynamo: bool = false,
            dynamo_params: dynamo_mod.Params = .{},
            bc_x: BcKind = .copy,
            bc_y: BcKind = .copy,
            bc_z: BcKind = .copy,
            specific_bc: ?SpecificBc = null,
            /// Opaque user context threaded verbatim to `specific_bc` on every
            /// ghost fill (e.g. which shock tube / which Michel solution the BC
            /// samples). Lets a runtime-parameterized BC avoid module-level
            /// state: the comptime-bound BCs (puffy) ignore it.
            bc_ctx: ?*const anyopaque = null,
            /// Worker threads for the per-step passes (P1: all of them run
            /// on the persistent team). 1 ≡ the serial path (bit-identical;
            /// the golden tests run here).
            nthreads: usize = 1,
            /// P4b node-width hardening: bind each team thread (main
            /// included) to one cpu of the process affinity mask; the
            /// cgroup cpuset under Slurm, so the binding stays inside the
            /// allocation. Linux only; inert elsewhere. No FP effect
            /// (scheduling never changes any written value).
            pin_threads: bool = false,
            /// MPI plan P4a: the communication backend (comm/comm.zig ;
            /// Serial no-ops by default, Mpi under -Dmpi). null ≡ serial
            /// semantics: when set, Sim binds the zero-copy exchange
            /// channels into `p` at init and runs the exchange episodes +
            /// end-of-step collective. Must outlive the Sim.
            comm: ?*comm_mod.Backend = null,
            /// The φ-slab decomposition this Sim's grid is the local piece
            /// of. null ≡ trivial (grid is global, all faces physical).
            decomp: ?comm_mod.Decomp = null,
        };

        allocator: std.mem.Allocator,
        grid: Grid,
        cache: precompute.MetricCache,
        opt: Options,
        /// P1: the persistent worker team all per-step passes dispatch on
        /// (threading.zig); null ≡ nthreads<=1 ≡ the serial path.
        team: ?*threading.Team,
        /// The resolved decomposition (opt.decomp orelse trivial). x/y faces
        /// are always physical; z faces stop being boundaries when ntz > 1
        /// (their ghosts come from the exchange; sim/bc.zig gates on this).
        decomp: comm_mod.Decomp,

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
        // Pass-owned scratch (redesign step 3): each pass module declares the
        // buffers it needs and Sim composes them; the optional ones exist
        // only when the option that uses them is on.
        /// The explicit operator's face stores (sim/explicit.zig): pbL/pbR,
        /// flL/flR per side and the combined flb. Flux-CT rewrites flb's B rows.
        faces: explicit_pass.Faces(NV),
        /// The fixup pass's whole-grid u/p backups (sim/fixup.zig, finite.c:5030).
        bak: fixup_pass.Backups(NV),

        scal: field_mod.Field(n_scal),
        flags: []i32,

        /// Constrained-transport scratch (sim/ct.zig: corner EMFs + the
        /// vector-potential work field). Present iff the config has MHD;
        /// `void` otherwise so a hydro-only Sim carries nothing.
        ct: CtScratch,
        /// Radiative-viscosity scratch (sim/rijvisc.zig: R^i_j per cell).
        /// Allocated only when opt.radviscosity.
        visc: ?rijvisc_mod.Scratch,
        /// Dynamo scratch (sim/dynamo.zig: ΔA_φ + the scale-height arrays).
        /// Allocated only when opt.dynamo.
        dynamo: ?dynamo_mod.Scratch,

        t: f64 = 0,
        /// C: global_time. Frozen at step start, used by set_bc.
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
        /// C: REDUCEORDERATBH. The largest radial cell index whose center is
        /// inside the BL horizon (reconstruction order is dropped by one for
        /// cell[0] ≤ this). std.math.minInt when disabled / no cell is inside,
        /// so the `cell[0] <= …` test is always false. Set once in init (r is
        /// monotonic in the radial index, so a single threshold suffices).
        reduce_order_ix_max: i64 = std.math.minInt(i64),
        /// implicit-solver diagnostics (C: global_int_slot counters), all
        /// run-cumulative; the driver deltas them for the per-step heartbeat.
        n_radimp_failures: u64 = 0,
        /// This step's implicit-failure count, MAXed across ranks by the
        /// end-of-step collective (the C abort path's load-bearing signal;
        /// == the local per-step delta on serial/1-rank runs).
        n_radimp_fail_step: u64 = 0,
        n_radimp_iters: u64 = 0,
        n_radimp_solves: u64 = 0,
        /// P0 per-pass wall-clock instrumentation (sim/timers.zig); always
        /// accumulating: the driver prints/resets it at its output cadence.
        timers: timers_mod.PassTimers = .{},

        /// The full-Field members allocated in init and freed in deinit, in one
        /// place so the two lists cannot drift (P5): the live p/u pair and the
        /// RK2IMEX stage buffers. Every other buffer is pass-owned scratch.
        const heap_field_names = .{
            "p",    "u",    "ut0",  "ut1",  "ut2",
            "dut1", "dut2", "drt1", "drt2", "uforget",
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
            // (6) MPI consistency: the grid must be the decomposition's local
            //     slab, the backend's ring must match its ntz, and a real
            //     split needs a backend (skipped z-face BCs would otherwise
            //     never be filled by anyone).
            const dc = opt.decomp orelse comm_mod.Decomp.serial(g);
            if (dc.local.nz != g.nz or dc.local.izoff != g.izoff) return error.InvalidConfig;
            if (dc.ntz > 1 and opt.comm == null) return error.InvalidConfig;
            if (opt.comm) |c| {
                if (c.size() != dc.ntz) return error.InvalidConfig;
            }
            // (7) correct_polaraxis is spherical-only, and only ONE half of it
            //     knows that: correctPolaraxis returns early on .mink while
            //     isCellCorrectedPolaraxis does not. The predicate would still
            //     claim the polar rows — u2p inverts B only, cell_fixup skips
            //     them "because they are overwritten later on", op_implicit
            //     skips them — while the overwrite that was to supply them
            //     never runs. p decouples from u and feeds back inward through
            //     reconstruction, with every fixup flag deliberately zeroed on
            //     that path, so nothing reports it. Reject outright (as (4)).
            if (opt.correct_polaraxis and cfg.coords == .mink) return error.InvalidConfig;

            var self: Self = undefined;
            self.allocator = allocator;
            self.grid = g;
            self.opt = opt;
            self.decomp = dc;
            self.n_radimp_fail_step = 0;
            self.t = 0;
            self.time = 0;
            self.dt = 0;
            self.own_dt = 0;
            self.tstepdenmax = 0;
            self.tstepdenmin = big;
            self.nstep = 0;
            // `self` starts as undefined, so field declaration defaults do
            // not run here.  Keep order reduction disabled unless the
            // explicit horizon scan below finds a qualifying cell.
            self.reduce_order_ix_max = std.math.minInt(i64);
            self.n_radimp_failures = 0;
            self.n_radimp_iters = 0;
            self.n_radimp_solves = 0;
            self.timers = .{};
            self.team = if (opt.nthreads > 1) try threading.Team.init(allocator, opt.nthreads, opt.pin_threads) else null;
            errdefer if (self.team) |tm| tm.deinit();

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
                // The team exists by now (spawned just above) and the cache
                // fills are pure per-cell writes — bit-identical threaded.
                .team = self.team,
            });
            errdefer self.cache.deinit();

            // The big per-cell stores are allocated raw and zeroed THROUGH
            // THE TEAM (P4b NUMA first-touch): the first write to each page
            // then happens on a worker, spreading the pages across the
            // node's memory controllers instead of concentrating them on
            // the main thread's. Zeros are zeros — bit-identical serially
            // (team == null degenerates to one whole-range memset).
            // Every allocation carries an errdefer, so a failed init leaks
            // nothing (redesign step 3; previously fatal-and-leak).
            var n_fields: usize = 0;
            errdefer {
                inline for (heap_field_names, 0..) |name, k| {
                    if (k < n_fields) @field(self, name).deinit();
                }
            }
            inline for (heap_field_names) |name| {
                @field(self, name) = try FieldT.initUninitialized(allocator, g);
                n_fields += 1;
                threading.parallelZero(self.team, @field(self, name).data);
            }
            // MPI plan §6.2: the four persistent zero-copy channels bind
            // directly into p's storage, which never moves after this.
            if (opt.comm) |c| try c.bindExchange(self.p.data, g, NV);
            errdefer if (opt.comm) |c| c.unbindExchange();

            self.faces = try explicit_pass.Faces(NV).init(allocator, g, self.team);
            errdefer self.faces.deinit();
            self.bak = try fixup_pass.Backups(NV).init(allocator, g, self.team);
            errdefer self.bak.deinit();
            self.scal = try field_mod.Field(n_scal).initUninitialized(allocator, g);
            errdefer self.scal.deinit();
            threading.parallelZero(self.team, self.scal.data);
            self.flags = try allocator.alloc(i32, g.cellCount() * n_flags);
            errdefer allocator.free(self.flags);
            @memset(self.flags, 0);
            if (comptime L.hasVar(.b1)) {
                self.ct = try ct.Scratch.init(allocator, g, self.team);
            } else {
                self.ct = {};
            }
            errdefer if (comptime L.hasVar(.b1)) self.ct.deinit();
            self.visc = if (opt.radviscosity) try rijvisc_mod.Scratch.init(allocator, g, self.team) else null;
            errdefer if (self.visc) |*v| v.deinit();
            self.dynamo = if (opt.dynamo) try dynamo_mod.Scratch.init(allocator, g, self.team) else null;
            errdefer if (self.dynamo) |*d| d.deinit();

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
            if (opt.comm) |c| {
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
            if (opt.reduceorderatbh) {
                const r_h = metric.rHorizonBL(opt.mp.a);
                const y_mid = g.yc(@intCast(g.ny / 2));
                var jx: i64 = -@as(i64, @intCast(g.ng));
                while (jx < @as(i64, @intCast(g.nx + g.ng))) : (jx += 1) {
                    const xx = [4]f64{ 0, g.xc(jx), y_mid, 0 };
                    const rbl = coco.cocoN(xx, opt.coords, .bl, opt.mp)[1];
                    if (rbl < r_h) self.reduce_order_ix_max = jx;
                }
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            const allocator = self.allocator;
            if (self.opt.comm) |c| c.unbindExchange();
            if (self.team) |tm| tm.deinit();
            inline for (heap_field_names) |name| {
                @field(self, name).deinit();
            }
            self.faces.deinit();
            self.bak.deinit();
            self.scal.deinit();
            allocator.free(self.flags);
            if (comptime L.hasVar(.b1)) self.ct.deinit();
            if (self.visc) |*v| v.deinit();
            if (self.dynamo) |*d| d.deinit();
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
            return self.grid.cellIndex(ix, iy, iz) * n_flags + @intFromEnum(f);
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
        pub fn isPhysicalBoundary(self: *const Self, face: BcFace) bool {
            return self.decomp.isPhysical(face);
        }

        /// One halo-exchange episode on the primitives (C: mpi_exchangedata).
        /// Zero-copy: Startall+Waitall over the persistent channels bound in
        /// init. No-op serially / at ntz==1. Main thread only (FUNNELED),
        /// which every call site satisfies (between team regions).
        pub fn exchangeHalos(self: *Self) void {
            if (self.opt.comm) |c| {
                self.timers.begin(.halo);
                defer self.timers.end();
                c.exchange();
            }
        }

        /// Fold one scalar to the global max (identity serially). Init-time
        /// use: PUFFY's BETANORM (§1.1-5).
        pub fn globalMax(self: *Self, v: f64) f64 {
            if (self.opt.comm) |c| {
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
            if (self.opt.comm) |c| {
                var buf = [1]f64{v};
                c.allreduceSum(buf[0..]);
                return buf[0];
            }
            return v;
        }

        /// The ONE per-step collective (plan §7.1), at the END of step() so
        /// the driver's next cflDt() reads global values (§1.1-6): a folded
        /// Allreduce(MAX) of [tstepdenmax, −tstepdenmin, this step's local
        /// implicit-failure count]. MAX is exactly associative → bitwise
        /// reproducible at any rank count.
        fn reduceStepGlobals(self: *Self, fail_before: u64) void {
            const local_fail = self.n_radimp_failures - fail_before;
            self.n_radimp_fail_step = local_fail;
            if (self.opt.comm) |c| {
                self.timers.begin(.collect);
                defer self.timers.end();
                var buf = [3]f64{ self.tstepdenmax, -self.tstepdenmin, @floatFromInt(local_fail) };
                c.allreduceMax(buf[0..]);
                self.tstepdenmax = buf[0];
                self.tstepdenmin = -buf[1];
                self.n_radimp_fail_step = @intFromFloat(buf[2]);
            }
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
        /// `pub` so the sim/-side passes (sim/rijvisc.zig) share the one
        /// definition of the corner convention with wavespeedRows.
        pub fn isCorner(self: *const Self, ix: i64, iy: i64, iz: i64) bool {
            var n: u8 = 0;
            if (ix < 0 or ix >= self.nxi()) n += 1;
            if (iy < 0 or iy >= self.nyi()) n += 1;
            if (iz < 0 or iz >= self.nzi()) n += 1;
            return n >= 2;
        }

        // ---- initialization -------------------------------------------------

        /// Set one domain cell's primitives and derived conserveds
        /// (C: PR_INIT body; pp then p2u).
        pub fn initCell(self: *Self, ix: i64, iy: i64, iz: i64, pp_in: [NV]f64) Error!void {
            var pp = pp_in;
            const geom = self.cache.fillGeometry(ix, iy, iz);
            const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
            self.p.store(ix, iy, iz, &pp);
            self.u.store(ix, iy, iz, &uu);
        }

        /// ko.c init tail + problem.c:59-82: halo exchange, ghost fill, and
        /// the initial timestep guess. Call after initCell over the domain
        /// (and after ct.calcBfromA for vector-potential problems). Uses
        /// `self.t` so a restart that has already adopted the checkpoint
        /// clock fills ghosts at the resumed time. `exchangeHalos` is a
        /// no-op serially / at 1 rank.
        pub fn finishInit(self: *Self) Error!void {
            self.exchangeHalos();
            try self.setBc(self.t, true);
            self.initTimestepGuess();
        }

        /// problem.c:59-82; initial dt guess from max_ws = 10⁴ (sim/timestep.zig).
        pub fn initTimestepGuess(self: *Self) void {
            timestep.initTimestepGuess(Self, self);
        }
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
            return timestep.calcWavespeeds(Self, self);
        }

        // ---- band-parallel dispatch ----------------------------------------
        // The persistent team + dynamic-tile mechanism (parallelRange +
        // ChunkResult) lives in threading.zig; the per-cell pass bodies live
        // in sim/*.zig (timestep, u2p, fixup, explicit, implicit_op, entropy,
        // stage), each generic over SimT. The operators below are the stable
        // entry points: they own the timers and hand the pass's banded worker
        // to the team. Most passes only ever report an error, so they go
        // through `parallelRangeErr`; the two that *reduce* — wavespeeds
        // (tsd_max/min) and the implicit operator (iteration/solve/failure
        // counters) — consume the merged `ChunkResult` themselves.

        /// C: calc_u2p (finite.c:546). Per-cell inversion + floors, then
        /// fixup averaging and a boundary refresh.
        /// `t` is the simulation time the refreshed ghosts are evaluated at
        /// (C: global_time). Every caller passes the current step's t, which
        /// equals self.time; taking it explicitly keeps the BC refresh from
        /// depending on ambient state a standalone caller (opExplicit,
        /// applyDynamo) might not have set.
        pub fn calcU2p(self: *Self, t: f64) Error!void {
            self.timers.begin(.u2p);
            defer self.timers.end();
            try threading.parallelRangeErr(Self, self, self.team, 0, self.nyi(), u2p_pass.rowsFn(Self));

            try self.cellFixup(.hd_fixup);
            if (comptime L.hasVar(.ee)) {
                try self.cellFixup(.rad_fixup);
            }
            try self.setBc(t, false);
        }

        /// C: cell_fixup (finite.c:5012), types FIXUP_U2PMHD / FIXUP_U2PRAD /
        /// FIXUP_RADIMP (sim/fixup.zig). This is the stable entry point: the
        /// enable switches and the timer; the flag scan, the u/p staging and
        /// the neighbour averaging live in the pass.
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
            return fixup_pass.cellFixup(Self, self, which);
        }

        // ---- the explicit operator ---------------------------------------------

        /// C: op_explicit (finite.c:633). The transport passes (sweep, flux
        /// combination, conserved update + metric source) live in
        /// sim/explicit.zig; this composes them in C's order.
        pub fn opExplicit(self: *Self, t: f64, dtin: f64) Error!void {
            // (upreexplicit/ppreexplicit copies skipped — see header)
            // calc_avgs_throughout: CALCHRONTHEGO only (M12)

            try self.calcWavespeeds();

            {
                self.timers.begin(.sweep);
                defer self.timers.end();
                inline for (0..3) |dim| try explicit_pass.sweep(Self, self, dim);
            }

            {
                self.timers.begin(.fluxes);
                defer self.timers.end();
                try explicit_pass.fluxesAtFaces(Self, self);
            }

            if (comptime cfg.has(.mhd)) {
                self.timers.begin(.fluxct);
                defer self.timers.end();
                ct.fluxCt(Self, self);
            }

            // conserved update: du from flux divergence + source terms
            {
                self.timers.begin(.update);
                defer self.timers.end();
                try explicit_pass.update(Self, self, dtin);
            }

            // (EXPLICIT_LAB_RAD_SOURCE not used — PUFFY couples implicitly)

            try self.calcU2p(t);
            // (MIXENTROPIESPROPERLY off)
        }

        // ---- stage arithmetic (exact C expression shapes; sim/stage.zig) ----------

        fn copyFull(self: *Self, dst: *FieldT, src: *const FieldT) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            stage.copyFull(Self, self, dst, src);
        }

        /// dst = (1/Δ)·a + (−1/Δ)·b over the domain (problem.c:181/230/…).
        fn stageDeriv(self: *Self, dst: *FieldT, a_f: *const FieldT, b_f: *const FieldT, delta: f64) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            stage.deriv(Self, self, dst, a_f, b_f, delta);
        }

        /// dst = a + f1·b + f2·c over the domain (problem.c:243/357).
        fn stageCombine(self: *Self, dst: *FieldT, a_f: *const FieldT, f1: f64, b_f: *const FieldT, f2: f64, c_f: *const FieldT) void {
            self.timers.begin(.stage);
            defer self.timers.end();
            stage.combine(Self, self, dst, a_f, f1, b_f, f2, c_f);
        }

        /// C: update_entropy (u2p.c:2257). Recompute S(ρ,u) and refresh the
        /// MHD conserveds. Band-parallel over iy (cell-local).
        pub fn updateEntropy(self: *Self) Error!void {
            self.timers.begin(.entropy);
            defer self.timers.end();
            try threading.parallelRangeErr(Self, self, self.team, 0, self.nyi(), entropy_pass.rowsFn(Self));
        }

        /// C: op_implicit (finite.c:1400). The implicit radiative source
        /// operator: per-cell solve_implicit_lab, RADIMPFIXUPFLAG, then
        /// cell_fixup(FIXUP_RADIMP). Structural no-op without radiation or
        /// when opt.opac is null (≡ SKIPRADSOURCE).
        pub fn opImplicit(self: *Self, t: f64, dtin: f64) Error!void {
            _ = t;
            if (comptime !L.hasVar(.ee)) return;
            _ = self.opt.opac orelse return;

            self.timers.begin(.implicit);
            defer self.timers.end();
            var ctx = implicit_op.Ctx(Self){ .sim = self, .dt = dtin };
            const res = threading.parallelRange(implicit_op.Ctx(Self), &ctx, self.team, 0, self.nyi(), implicit_op.rowsWorker(Self));
            self.n_radimp_failures += res.n_fail;
            self.n_radimp_iters += res.n_iters;
            self.n_radimp_solves += res.n_solves;
            if (res.err) |e| return e;

            try self.cellFixup(.radimp_fixup);
        }

        /// C: do_correct (finite.c:594); CORRECT_POLARAXIS only; the 3D /
        /// smoothing / NS-surface variants are not PUFFY machinery. The
        /// overwrite, the band predicate and (later) the polar-axis EMF
        /// zeroing live in sim/polaraxis.zig; this is the stable method entry
        /// point that problems and tests call.
        pub fn doCorrect(self: *Self) Error!void {
            if (polaraxis.band(Self, self) == null) return;
            self.timers.begin(.correct);
            defer self.timers.end();
            return polaraxis.correct(Self, self);
        }

        // ---- one full RK2IMEX step (problem.c:141-402) ----------------------------

        pub fn step(self: *Self, forced_dt: ?f64) Error!void {
            comptime std.debug.assert(cfg.timestepping == .rk2imex);

            self.timers.begin(.step);
            defer self.timers.end();

            const t = self.t;
            self.time = t; // C: global_time = t (write-only: kept to mirror C; calcU2p now takes t explicitly)

            // The CFL denominator must have been seeded (initTimestepGuess, part
            // of C's solve preamble) or be forced — otherwise tstepdenmax is 0
            // and own_dt is +inf → NaNs on the first step, diagnosed only far
            // downstream. Turn that ordering bug into an immediate failure.
            std.debug.assert(forced_dt != null or self.tstepdenmax > 0);
            self.own_dt = self.cflDt();
            const dt = forced_dt orelse self.own_dt;
            self.dt = dt;
            // Baseline for the end-of-step failure fold (§7.1): the counter
            // is run-cumulative, the collective reduces this step's delta.
            const radimp_fail_at_step_start = self.n_radimp_failures;

            // reset wavespeed accumulators for this step
            self.tstepdenmax = -1;
            self.tstepdenmin = big;

            // C: calc_Rij_visc_total (problem.c:127) — once per step, with
            // global_dt = this step's dt, feeding the RADVISCNUDAMP cap.
            if (self.opt.radviscosity and comptime L.hasVar(.ee)) {
                self.timers.begin(.radvisc);
                defer self.timers.end();
                try rijvisc_mod.calcRijViscTotal(Self, self, dt);
            }

            const gam_imex = 1.0 - 1.0 / @sqrt(2.0);

            // my_finger: PR_FINGER not defined for any target problem
            timestep.saveTimesteps(Self, self);
            // set_gammagas: CONSISTENTGAMMA off

            // ---- 1st implicit ----
            self.copyFull(&self.ut0, &self.u);
            // (C copies p→ptm1 and, post-implicit, p→ppostimplicit here; both
            //  are write-only on this path — no Zig consumer — so dropped.)
            try self.opImplicit(t, dt * gam_imex); // U(1)
            self.stageDeriv(&self.drt1, &self.u, &self.ut0, dt * gam_imex);

            // ---- 1st explicit ----
            self.copyFull(&self.ut1, &self.u);
            try self.calcU2p(t);
            try self.doCorrect();
            self.exchangeHalos(); // C: mpi_exchangedata BEFORE set_bc (problem.c:198-204)
            try self.setBc(t, false);
            try self.opExplicit(t, dt); // F(U(1))
            if (self.opt.dynamo and comptime L.hasVar(.b1)) {
                self.timers.begin(.dynamo);
                defer self.timers.end();
                try dynamo_mod.applyDynamo(Self, self, t, dt);
            }
            // op_intermediate (electrons) — no-op
            entropy_pass.copyEntropyCount(Self, self);
            self.stageDeriv(&self.dut1, &self.u, &self.ut1, dt);

            // ---- 1st together: u = ut0 + dt·dut1 + dt(1−2γ)·drt1 ----
            self.stageCombine(&self.u, &self.ut0, dt, &self.dut1, dt * (1.0 - 2.0 * gam_imex), &self.drt1);

            // ---- 2nd implicit ----
            self.copyFull(&self.uforget, &self.u);
            try self.calcU2p(t);
            // (C copies p→ptm1 / p→ppostimplicit here; write-only, dropped.)
            try self.doCorrect();
            try self.opImplicit(t, gam_imex * dt); // U(2)
            self.stageDeriv(&self.drt2, &self.u, &self.uforget, dt * gam_imex);

            // ---- 2nd explicit ----
            self.copyFull(&self.ut2, &self.u);
            try self.calcU2p(t);
            try self.doCorrect();
            self.exchangeHalos(); // C: problem.c:314-320
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
            try self.calcU2p(t);
            try self.doCorrect();
            // C: problem.c:386-392 — the THIRD per-step exchange site, and
            // not optional. `calcRijViscTotal` runs at the TOP of the next
            // step (before that step's first exchange) and reads z-ghost
            // primitives over iz ∈ [−1, nz+1); without this the radiative
            // shear viscosity would consume ghosts a full RK stage stale.
            self.exchangeHalos();
            try self.setBc(t, false);

            self.t = t + dt;
            try self.updateEntropy();
            self.reduceStepGlobals(radimp_fail_at_step_start);
            self.nstep += 1;
        }
    };
}

test {
    _ = FaceStore(5);
    _ = timers_mod;
}
