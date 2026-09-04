//! Wavespeeds and the CFL timestep, generic over CoreT (the house
//! `fn f(comptime CoreT: type, sim: *CoreT)` pattern, like sim/bc.zig):
//!
//!   calcWavespeeds:    C: calc_wavespeeds (finite.c:356) over domain + 1
//!                      ghost layer; band-parallel over iy with the CFL
//!                      denominator max/min reduced through ChunkResult.
//!   saveWavespeeds:    C: save_wavespeeds (finite.c:394).
//!   saveTimesteps:     C: save_timesteps (finite.c:490), no SHORTERTIMESTEP.
//!   initTimestepGuess: problem.c:59-82; initial dt guess from max_ws = 10⁴.
//!
//! The CFL denominator per cell is Σ_d ws_d/Δx_d divided by tsteplim, the
//! per-dimension speeds SUMMED (C's conservative multi-D bound), and dt is
//! 1/max over the domain. Gate: tests/timestep_tests.zig pins Core.cflDt to
//! that bound with the SR hydro eigenvalues (velocity addition along a
//! boost, c_s√(1−v²)/√(1−v²c_s²) transverse) at 1e-12.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04); the
//! bodies, loop orders and expression shapes are unchanged.

const relele = @import("../relele.zig");
const wavespeeds = @import("../physics/wavespeeds.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const threading = @import("../threading.zig");
const storage = @import("storage.zig");

const Error = relele.Error || error{OutOfMemory};

/// problem.c:59-82; initial dt guess from max_ws = 10⁴.
pub fn initTimestepGuess(comptime CoreT: type, self: *CoreT) void {
    const ws: f64 = 10000.0;
    var tsd: f64 = undefined;
    if (self.grid.nz > 1) {
        tsd = ws / self.min_dx + ws / self.min_dy + ws / self.min_dz;
    } else if (self.grid.ny > 1) {
        tsd = ws / self.min_dx + ws / self.min_dy;
    } else {
        tsd = ws / self.min_dx;
    }
    tsd /= self.num.tsteplim;
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

/// C: calc_wavespeeds (finite.c:356) over domain + 1 ghost layer.
/// Band-parallel over iy; the CFL-denominator max/min reduce runs as
/// per-worker partials merged after the region (max/min are
/// order-insensitive, so this is bit-identical to the serial scan).
pub fn calcWavespeeds(comptime CoreT: type, self: *CoreT) Error!void {
    const ly: i64 = if (self.grid.ny > 1) 1 else 0;
    const res = threading.parallelRange(CoreT, self, self.team, -ly, self.nyi() + ly, rowsWorker(CoreT));
    if (res.err) |e| return e;
    if (res.tsd_max > self.tstepdenmax) self.tstepdenmax = res.tsd_max;
    if (res.tsd_min < self.tstepdenmin) self.tstepdenmin = res.tsd_min;
}

fn rowsWorker(comptime CoreT: type) fn (*CoreT, i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(self: *CoreT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            wavespeedRows(CoreT, self, iy0, iy1, res) catch |e| {
                res.err = e;
            };
        }
    }.w;
}

/// The per-cell wavespeed body for iy ∈ [iy0, iy1) (all iz, all ix
/// incl. the ±1 ghost layer).
fn wavespeedRows(comptime CoreT: type, self: *CoreT, iy0: i64, iy1: i64, res: *threading.ChunkResult) Error!void {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    const active_dims = [3]bool{ self.grid.nx > 1, self.grid.ny > 1, self.grid.nz > 1 };
    const lx: i64 = if (active_dims[0]) 1 else 0;
    const lz: i64 = if (active_dims[2]) 1 else 0;

    var iz: i64 = -lz;
    while (iz < self.nzi() + lz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = -lx;
            while (ix < self.nxi() + lx) : (ix += 1) {
                if (comptime !CoreT.wide) {
                    if (self.isCorner(ix, iy, iz)) continue;
                }
                var pp: [NV]f64 = undefined;
                self.p.load(ix, iy, iz, &pp);
                const geom = self.cache.fillGeometry(ix, iy, iz);
                const aaa = try wavespeeds.gasWavespeedsLr(cfg, pp, &geom, self.phys.gam, active_dims);

                var rad0: [6]f64 = @splat(0);
                var radl: [6]f64 = @splat(0);
                if (comptime L.hasVar(.ee)) {
                    // optical depth over the cell (physics.c:549-558);
                    // ix could be a face index in C, hence the max of
                    // left/right sizes
                    var tautot: [3]f64 = @splat(0);
                    if (self.phys.opac) |*op| {
                        const chi = try radforce.calcChiSlim(cfg, pp, &geom, self.phys.gam, op);
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
                    // C: DAMPRADWAVESPEEDNEARAXIS (rad.c:3690) — within N
                    // cells of either pole, discard the optical-depth
                    // damping so the radiative wavespeed keeps its
                    // undamped value 1/3 (raising near-axis diffusion for
                    // stability). Zeroing tautot is equivalent: it drives
                    // calcRadWavespeeds' tautot≤0 branch → rv²=1/3.
                    if (self.num.dampradwavespeednearaxis > 0) {
                        const nc: i64 = @intCast(self.num.dampradwavespeednearaxis);
                        const ny: i64 = @intCast(self.grid.ny);
                        if (iy < nc or iy >= ny - nc) tautot = @splat(0);
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
                saveWavespeeds(CoreT, self, ix, iy, iz, aaa, rad0, radl, res);
            }
        }
    }
}

/// C: save_wavespeeds (finite.c:394). Store per-cell speeds and,
/// for domain cells, the CFL denominator (feeding the next dt).
/// rad0 (unlimited by τ) only enters the timestep; radl (τ-limited)
/// is what the flux combination reads. The dt max/min go into the
/// worker's ChunkResult partials (merged after the region).
fn saveWavespeeds(comptime CoreT: type, self: *CoreT, ix: i64, iy: i64, iz: i64, aaa: [6]f64, rad0: [6]f64, radl: [6]f64, res: *threading.ChunkResult) void {
    const L = CoreT.Layout;
    for (0..3) |d| {
        self.scSet(storage.ahd_l[d], ix, iy, iz, aaa[2 * d]);
        self.scSet(storage.ahd_r[d], ix, iy, iz, aaa[2 * d + 1]);
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
            self.scSet(storage.arad_l[d], ix, iy, iz, radl[2 * d]);
            self.scSet(storage.arad_r[d], ix, iy, iz, radl[2 * d + 1]);
            self.scSet(storage.arad_m[d], ix, iy, iz, @max(@abs(radl[2 * d]), @abs(radl[2 * d + 1])));
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
    tsd /= self.num.tsteplim;
    self.scSet(.tstepden, ix, iy, iz, tsd);
    if (tsd > res.tsd_max) res.tsd_max = tsd;
    if (tsd < res.tsd_min) res.tsd_min = tsd;
}

/// C: save_timesteps (finite.c:490), no SHORTERTIMESTEP.
pub fn saveTimesteps(comptime CoreT: type, self: *CoreT) void {
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
