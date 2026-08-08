//! Open MPI ABI. Handles are struct pointers; the predefined objects are
//! extern global symbols in libmpi (`MPI_COMM_WORLD` expands to
//! `&ompi_mpi_comm_world` — a LINK-TIME value, which is exactly why it
//! cannot be a Zig comptime constant and lives behind a function here;
//! MPI plan §3.1/§3.2). The symbols are declared as `u8` because only
//! their addresses are ever used — the pointee is opaque to us.

pub const RawComm = ?*anyopaque;
pub const RawDatatype = ?*anyopaque;
pub const RawOp = ?*anyopaque;
pub const RawRequest = ?*anyopaque;
/// `typedef struct ompi_file_t *MPI_File;` (mpi.h:438, verified 5.0.9).
pub const RawFile = ?*anyopaque;
pub const RawInfo = ?*anyopaque;

/// Open MPI MPI_Status: 24 bytes on LP64, MPI_SOURCE at offset 0.
pub const Status = extern struct {
    source: c_int,
    tag: c_int,
    err: c_int,
    cancelled: c_int,
    ucount: usize,
};

pub const request_null: RawRequest = null;

extern var ompi_mpi_comm_world: u8;
extern var ompi_mpi_double: u8;
extern var ompi_mpi_byte: u8;
extern var ompi_mpi_op_max: u8;
extern var ompi_mpi_op_min: u8;
extern var ompi_mpi_op_sum: u8;
extern var ompi_mpi_info_null: u8;

pub fn commWorld() RawComm {
    return @ptrCast(&ompi_mpi_comm_world);
}
pub fn dtDouble() RawDatatype {
    return @ptrCast(&ompi_mpi_double);
}
pub fn opMax() RawOp {
    return @ptrCast(&ompi_mpi_op_max);
}
pub fn opMin() RawOp {
    return @ptrCast(&ompi_mpi_op_min);
}
pub fn opSum() RawOp {
    return @ptrCast(&ompi_mpi_op_sum);
}
pub fn dtByte() RawDatatype {
    return @ptrCast(&ompi_mpi_byte);
}
pub fn infoNull() RawInfo {
    return @ptrCast(&ompi_mpi_info_null);
}

/// MPI_STATUSES_IGNORE == NULL in Open MPI. The `align(1)` matches the
/// MPICH family's signature (which needs it — see abi_mpich.zig) so
/// core.zig's extern declaration is family-invariant.
pub fn statusesIgnore() ?[*]align(1) Status {
    return null;
}

/// MPI_STATUS_IGNORE == NULL, like statusesIgnore (`align(1)` for the
/// family-invariant core.zig signature — MPICH's sentinel is address 1).
pub fn statusIgnore() ?*align(1) Status {
    return null;
}

/// MPI_IN_PLACE == (void*)1 in Open MPI.
pub fn inPlace() *const anyopaque {
    return @ptrFromInt(1);
}
