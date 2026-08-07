//! MPICH-family ABI (MPICH, Intel MPI, MVAPICH, Cray MPICH — the
//! `libmpi.so.12` compatibility initiative). Handles are `int`; predefined
//! objects are compile-time hex constants. Values verified against MPICH
//! 4.x `mpi.h` (MPI plan §3.1); the build's `mpi-abi-check` self-test
//! (plan §3.2) cross-checks sizeof/offsetof/sentinels against the found
//! header at configure time on cluster CI.

pub const RawComm = c_int;
pub const RawDatatype = c_int;
pub const RawOp = c_int;
pub const RawRequest = c_int;

/// MPICH MPI_Status: 20 bytes, MPI_SOURCE at offset 8.
pub const Status = extern struct {
    count_lo: c_int,
    count_hi_and_cancelled: c_int,
    source: c_int,
    tag: c_int,
    err: c_int,
};

pub const request_null: RawRequest = 0x2c000000;

pub fn commWorld() RawComm {
    return 0x44000000;
}
pub fn dtDouble() RawDatatype {
    return 0x4c00080b;
}
pub fn opMax() RawOp {
    return 0x58000001;
}
pub fn opMin() RawOp {
    return 0x58000002;
}
pub fn opSum() RawOp {
    return 0x58000003;
}

/// MPI_STATUSES_IGNORE == (MPI_Status*)1 — a sentinel, never a Zig optional
/// (plan §3.1: the two families disagree on the bit pattern). `align(1)` is
/// load-bearing: `Status` is five c_int, so its natural alignment is 4 and
/// Zig rejects the address 1 without it. The pointer is never dereferenced
/// (MPI reads the value as a sentinel), so under-aligning it is sound.
pub fn statusesIgnore() ?[*]align(1) Status {
    return @ptrFromInt(1);
}

/// MPI_IN_PLACE == (void*)-1.
pub fn inPlace() *const anyopaque {
    return @ptrFromInt(@as(usize, @bitCast(@as(isize, -1))));
}
