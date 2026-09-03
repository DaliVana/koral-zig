//! koral: a Zig reimplementation of KORAL (GR radiation MHD).
//!
//! `koral` is a library; a *problem* is an executable that imports it,
//! declares a comptime `Config`, loads runtime `Params`, and supplies
//! init/boundary hooks. See koral_lite/docs/zig-rewrite-architecture.md.

const std = @import("std");

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const Module = config.Module;

pub const layout = @import("layout.zig");
pub const VarLayout = layout.VarLayout;
pub const VarTag = layout.VarTag;

pub const units = @import("units.zig");
pub const Units = units.Units;

pub const grid = @import("grid.zig");
pub const Grid = grid.Grid;

pub const field = @import("field.zig");
pub const Field = field.Field;

pub const params = @import("params.zig");
pub const Params = params.Params;

pub const geometry = @import("geometry.zig");
pub const Geometry = geometry.Geometry;

pub const math = struct {
    pub const dual = @import("math/dual.zig");
    pub const Dual = dual.Dual;
    pub const Dual3 = dual.Dual3;
    pub const quad = @import("math/quad.zig");
    pub const linalg = @import("math/linalg.zig");
    pub const misc = @import("math/misc.zig");
    pub const simd = @import("math/simd.zig");
};

pub const metric = struct {
    pub const core = @import("metric/metric.zig");
    pub const forms = @import("metric/forms.zig");
    pub const coco = @import("metric/coco.zig");
    pub const precompute = @import("metric/precompute.zig");
    pub const MetricParams = core.MetricParams;
    pub const CoordData = core.CoordData;
    pub const MetricCache = precompute.MetricCache;
};

pub const relele = @import("relele.zig");
pub const frames = @import("frames.zig");
pub const p2u = @import("p2u.zig");

pub const physics = struct {
    pub const mhd = @import("physics/bfield.zig");
    pub const hydro = @import("physics/hydro.zig");
    pub const radiation = @import("physics/radiation.zig");
    pub const wavespeeds = @import("physics/wavespeeds.zig");
    pub const flux = @import("physics/flux.zig");
    pub const thermo = @import("physics/thermo.zig");
    pub const opacities = @import("physics/opacities.zig");
    pub const mesa = @import("physics/mesa.zig");
    pub const radforce = @import("physics/radforce.zig");
    pub const radvisc = @import("physics/radvisc.zig");
};

pub const solve = struct {
    pub const invert = @import("solve/invert.zig");
    pub const invert_rad = @import("solve/invert_rad.zig");
    pub const implicit = @import("solve/implicit.zig");
};

pub const fv = struct {
    pub const recon = @import("fv/recon.zig");
    pub const laxf = @import("fv/laxf.zig");
};

pub const sim = @import("sim.zig");
pub const Sim = sim.Sim;

pub const magn = struct {
    pub const ct = @import("sim/ct.zig");
    pub const dynamo = @import("sim/dynamo.zig");
};

pub const problems = struct {
    pub const puffy = @import("problems/puffy/puffy.zig");
};

pub const testing = struct {
    pub const golden = @import("testing/golden.zig");
    pub const tubes = @import("testing/tubes.zig");
    /// Zig-generated regression baselines; NOT the C oracle. See the module
    /// header for what a mismatch does and does not mean.
    pub const selfgolden = @import("testing/selfgolden.zig");
    pub const selfscenarios = @import("testing/selfscenarios.zig");
};

pub const comm = struct {
    pub const seam = @import("comm/comm.zig");
    /// The comptime-selected backend: Serial by default, Mpi under -Dmpi.
    pub const Comm = seam.Backend;
    pub const enabled = seam.enabled;
    pub const serial = @import("comm/serial.zig");
    pub const Serial = serial.Serial;
    pub const decomp = @import("comm/decomp.zig");
    pub const Decomp = decomp.Decomp;
    pub const decompose = decomp.decompose;
};

pub const io = struct {
    pub const scalars = @import("io/scalars.zig");
    pub const dump = @import("io/dump.zig");
    pub const silo = @import("io/silo.zig");
};

pub const render = @import("render/render.zig");

test {
    // Pull in all module tests. Test files live apart from the code they
    // gate, one folder per family:
    //   * theory gates (math/analytic identities):  koral/tests/<subsystem>_tests.zig
    //   * C-oracle goldens (tests/golden/*.kgld):   koral/tests/golden/<subsystem>_golden_tests.zig
    // New test files must be added to the list below (there is no auto-registration).
    _ = config;
    _ = layout;
    _ = units;
    _ = grid;
    _ = field;
    _ = params;
    _ = comm.serial;
    _ = comm.decomp;
    _ = geometry;
    _ = math.dual;
    _ = math.linalg;
    _ = math.simd;
    _ = metric.core;
    _ = metric.forms;
    _ = metric.coco;
    _ = metric.precompute;
    _ = relele;
    _ = frames;
    _ = p2u;
    _ = physics.mhd;
    _ = physics.hydro;
    _ = solve.invert;
    _ = physics.radiation;
    _ = physics.wavespeeds;
    _ = physics.flux;
    _ = physics.thermo;
    _ = physics.opacities;
    _ = physics.mesa;
    _ = physics.radforce;
    _ = physics.radvisc;
    _ = fv.recon;
    _ = fv.laxf;
    _ = sim;
    _ = magn.ct;
    _ = magn.dynamo;
    _ = solve.invert_rad;
    _ = solve.implicit;
    _ = io.scalars;
    _ = io.dump;
    _ = io.silo;
    _ = render;
    _ = render.image;
    _ = render.series;
    _ = render.sweep;
    _ = render.adaptive;
    _ = render.fits;
    _ = render.verify;
    _ = @import("math/quad.zig");
    _ = @import("problems/puffy/puffy.zig");
    _ = @import("testing/tubes.zig");

    // theory gates — koral/tests/
    _ = @import("tests/dynamo_tests.zig");
    _ = @import("tests/evolution_tests.zig");
    _ = @import("tests/flux_tests.zig");
    _ = @import("tests/implicit_tests.zig");
    _ = @import("tests/metric_tests.zig");
    _ = @import("tests/mhd_evolution_tests.zig");
    _ = @import("tests/mpi_abi_tests.zig");
    _ = @import("tests/opacity_tests.zig");
    _ = @import("tests/paper2013_hydro_tests.zig");
    _ = @import("tests/paper2013_pulse_tests.zig");
    _ = @import("tests/polaraxis_tests.zig");
    _ = @import("tests/puffy_tests.zig");
    _ = @import("tests/radiation_tests.zig");
    _ = @import("tests/radstep_tests.zig");
    _ = @import("tests/radtube_tests.zig");
    _ = @import("tests/radvisc_tests.zig");
    _ = @import("tests/render_tests.zig");
    _ = @import("tests/restart_tests.zig");
    _ = @import("tests/scalars_tests.zig");
    _ = @import("tests/selfgolden_tests.zig");
    _ = @import("tests/sim_tests.zig");
    _ = @import("tests/simd_tests.zig");
    _ = @import("tests/state_tests.zig");
    _ = @import("tests/threading_tests.zig");

    // C-oracle goldens — koral/tests/golden/
    _ = @import("tests/golden/dynamo_golden_tests.zig");
    _ = @import("tests/golden/flux_golden_tests.zig");
    _ = @import("tests/golden/implicit_golden_tests.zig");
    _ = @import("tests/golden/metric_golden_tests.zig");
    _ = @import("tests/golden/opac_golden_tests.zig");
    _ = @import("tests/golden/puffy_golden_tests.zig");
    _ = @import("tests/golden/puffystep_golden_tests.zig");
    _ = @import("tests/golden/rad_golden_tests.zig");
    _ = @import("tests/golden/state_golden_tests.zig");
    _ = @import("tests/golden/step_golden_tests.zig");
    _ = @import("tests/golden/visc_golden_tests.zig");
}
