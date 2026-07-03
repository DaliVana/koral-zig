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
}
