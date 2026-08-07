//! Comptime ABI-family selection (MPI plan §3.2): `build.zig` probes the
//! system MPI (or honors `-Dmpi-family`) and bakes the choice into
//! `build_options.mpi_family`. The family types/constants never leak past
//! `comm/mpi/` — the backend exposes only wrapper methods. `abi_std.zig`
//! (the MPI-5 standard ABI) joins as a third family once facility modules
//! ship it (plan §11.1-5).

const std = @import("std");
const build_options = @import("build_options");

pub const Family = enum { mpich, ompi };

pub const family: Family = blk: {
    const s = build_options.mpi_family;
    if (std.mem.eql(u8, s, "mpich")) break :blk .mpich;
    if (std.mem.eql(u8, s, "ompi")) break :blk .ompi;
    @compileError("comm/mpi: unknown or missing mpi_family '" ++ s ++
        "' (build with -Dmpi so build.zig probes mpicc, or set -Dmpi-family=mpich|ompi)");
};

const impl = switch (family) {
    .mpich => @import("abi_mpich.zig"),
    .ompi => @import("abi_ompi.zig"),
};

pub const RawComm = impl.RawComm;
pub const RawDatatype = impl.RawDatatype;
pub const RawOp = impl.RawOp;
pub const RawRequest = impl.RawRequest;
pub const Status = impl.Status;
pub const request_null = impl.request_null;
pub const commWorld = impl.commWorld;
pub const dtDouble = impl.dtDouble;
pub const opMax = impl.opMax;
pub const opMin = impl.opMin;
pub const opSum = impl.opSum;
pub const statusesIgnore = impl.statusesIgnore;
pub const inPlace = impl.inPlace;

// Family-invariant constants (identical in MPICH and Open MPI).
pub const success: c_int = 0;
pub const thread_single: c_int = 0;
pub const thread_funneled: c_int = 1;
pub const thread_serialized: c_int = 2;
pub const thread_multiple: c_int = 3;
