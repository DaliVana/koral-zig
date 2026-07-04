//! koral — a Zig reimplementation of KORAL (GR radiation MHD).
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

pub const state = @import("state.zig");
pub const State = state.State;

pub const geometry = @import("geometry.zig");
pub const Geometry = geometry.Geometry;

pub const math = struct {
    pub const dual = @import("math/dual.zig");
    pub const Dual3 = dual.Dual3;
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
    pub const mhd = @import("physics/mhd.zig");
    pub const hydro = @import("physics/hydro.zig");
    pub const radiation = @import("physics/radiation.zig");
    pub const wavespeeds = @import("physics/wavespeeds.zig");
    pub const flux = @import("physics/flux.zig");
};

pub const solve = struct {
    pub const invert = @import("solve/invert.zig");
    pub const invert_rad = @import("solve/invert_rad.zig");
};

pub const recon = @import("recon/recon.zig");

pub const riemann = struct {
    pub const laxf = @import("flux/laxf.zig");
};

pub const sim = @import("sim.zig");
pub const Sim = sim.Sim;

pub const magn = struct {
    pub const ct = @import("magn/ct.zig");
};

pub const comm = struct {
    pub const serial = @import("comm/serial.zig");
    pub const Serial = serial.Serial;
    pub const ReduceOp = serial.ReduceOp;
};

test {
    // Pull in all module tests.
    _ = config;
    _ = layout;
    _ = units;
    _ = grid;
    _ = field;
    _ = params;
    _ = state;
    _ = comm.serial;
    _ = geometry;
    _ = math.dual;
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
    _ = recon;
    _ = riemann.laxf;
    _ = sim;
    _ = magn.ct;
    _ = solve.invert_rad;
    _ = @import("metric/tests.zig");
    _ = @import("evolution_tests.zig");
    _ = @import("mhd_evolution_tests.zig");
    _ = @import("radiation_tests.zig");
    _ = @import("state_tests.zig");
    _ = @import("flux_tests.zig");
    _ = @import("golden_test.zig");
    _ = @import("golden_state_test.zig");
    _ = @import("golden_flux_test.zig");
    _ = @import("golden_rad_test.zig");
    _ = @import("golden_step_test.zig");
}
