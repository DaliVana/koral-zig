//! MPI ABI self-check (MPI plan §3.2) — the guard that was missing.
//!
//! `comm/mpi/abi.zig` picks ONE family at comptime, so a serial build (and
//! the Open MPI dev loop) never semantically analyzes the other family's
//! file. That is exactly how `abi_mpich.zig` came to contain a hard
//! compile error — `@ptrFromInt(1)` into a 4-aligned `?[*]Status` — that
//! nothing in the tree could surface until someone built on an Intel-MPI
//! or Cray cluster.
//!
//! These tests import BOTH family files directly and call their pure
//! (non-extern) accessors, so `zig build test` — with no `-Dmpi`, no
//! libmpi, on any machine — fails if either family stops compiling or its
//! sentinel/layout table drifts.
//!
//! Deliberately NOT covered here: `commWorld/dtDouble/op*` of the Open MPI
//! family, which take the address of `extern var ompi_mpi_*` symbols.
//! Calling those would make the serial test artifact link against libmpi,
//! breaking the "a serial build never references MPI" contract. Their
//! *signatures* are still checked below via @TypeOf, which needs no symbol.

const std = @import("std");
const mpich = @import("comm/mpi/abi_mpich.zig");
const ompi = @import("comm/mpi/abi_ompi.zig");

test "MPICH ABI: sentinels are representable and carry the documented values" {
    // The regression this file exists for: MPI_STATUSES_IGNORE is the
    // address 1, which cannot be held by a pointer type that requires
    // 4-byte alignment. Calling the function is what forces analysis.
    const si = mpich.statusesIgnore();
    try std.testing.expectEqual(@as(usize, 1), @intFromPtr(si.?));
    // MPI_IN_PLACE == (void*)-1 in the MPICH family.
    try std.testing.expectEqual(
        @as(usize, @bitCast(@as(isize, -1))),
        @intFromPtr(mpich.inPlace()),
    );
    // Predefined handles are comptime hex constants (mpich mpi.h).
    try std.testing.expectEqual(@as(c_int, 0x44000000), mpich.commWorld());
    try std.testing.expectEqual(@as(c_int, 0x4c00080b), mpich.dtDouble());
    try std.testing.expectEqual(@as(c_int, 0x58000001), mpich.opMax());
    try std.testing.expectEqual(@as(c_int, 0x58000002), mpich.opMin());
    try std.testing.expectEqual(@as(c_int, 0x58000003), mpich.opSum());
    try std.testing.expectEqual(@as(c_int, 0x2c000000), mpich.request_null);
}

test "MPICH ABI: MPI_Status is 20 bytes with MPI_SOURCE at offset 8" {
    // Plan §3.1's verified layout table. A drift here silently corrupts
    // every status-returning call.
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(mpich.Status));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(mpich.Status, "source"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(mpich.Status, "tag"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(mpich.Status, "err"));
    try std.testing.expectEqual(c_int, mpich.RawComm);
}

test "Open MPI ABI: sentinels are representable and carry the documented values" {
    // MPI_STATUSES_IGNORE == NULL here — the reason this family never hit
    // the alignment error and the MPICH one went unnoticed.
    try std.testing.expectEqual(@as(?[*]align(1) ompi.Status, null), ompi.statusesIgnore());
    // MPI_IN_PLACE == (void*)1 in Open MPI (NOT -1 — the families differ).
    try std.testing.expectEqual(@as(usize, 1), @intFromPtr(ompi.inPlace()));
    try std.testing.expectEqual(@as(ompi.RawRequest, null), ompi.request_null);
}

test "Open MPI ABI: MPI_Status is 24 bytes with MPI_SOURCE at offset 0" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(ompi.Status));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ompi.Status, "source"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(ompi.Status, "tag"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(ompi.Status, "err"));
    // Handles are opaque pointers, not ints.
    try std.testing.expectEqual(@sizeOf(usize), @sizeOf(ompi.RawComm));
}

test "the two families are genuinely different ABIs" {
    // Guards against 'fixing' one family by copying the other: the whole
    // point of the two-file design is that these disagree.
    try std.testing.expect(@sizeOf(mpich.Status) != @sizeOf(ompi.Status));
    try std.testing.expect(@offsetOf(mpich.Status, "source") != @offsetOf(ompi.Status, "source"));
    try std.testing.expect(@intFromPtr(mpich.inPlace()) != @intFromPtr(ompi.inPlace()));
    try std.testing.expect(mpich.RawComm != ompi.RawComm);
}

test "MPI-IO ABI: file handles, info handles, and sentinels (both families)" {
    // P4b additions (plan §8.1). MPI_File is a POINTER in both families —
    // the one MPICH handle that is not an int (ROMIO's ADIOI_FileD*).
    try std.testing.expectEqual(?*anyopaque, mpich.RawFile);
    try std.testing.expectEqual(?*anyopaque, ompi.RawFile);
    // MPI_Info stays on the family pattern (int vs pointer).
    try std.testing.expectEqual(c_int, mpich.RawInfo);
    try std.testing.expectEqual(?*anyopaque, ompi.RawInfo);
    try std.testing.expectEqual(@as(c_int, 0x1c000000), mpich.infoNull());
    // MPI_BYTE (MPICH hex constant; OMPI's is an extern symbol — signature
    // checked below, never called here, keeping the artifact libmpi-free).
    try std.testing.expectEqual(@as(c_int, 0x4c00010d), mpich.dtByte());
    // MPI_STATUS_IGNORE: address 1 vs NULL — the same split (and the same
    // align(1) requirement) as STATUSES_IGNORE.
    try std.testing.expectEqual(@as(usize, 1), @intFromPtr(mpich.statusIgnore().?));
    try std.testing.expectEqual(@as(?*align(1) ompi.Status, null), ompi.statusIgnore());
    try std.testing.expectEqual(
        ?*align(1) ompi.Status,
        @typeInfo(@TypeOf(ompi.statusIgnore)).@"fn".return_type.?,
    );
    try std.testing.expectEqual(ompi.RawDatatype, @typeInfo(@TypeOf(ompi.dtByte)).@"fn".return_type.?);
    try std.testing.expectEqual(ompi.RawInfo, @typeInfo(@TypeOf(ompi.infoNull)).@"fn".return_type.?);
}

test "both families satisfy the signatures core.zig declares" {
    // Signature-only checks — no call, so Open MPI's extern predefined
    // symbols are never referenced and the serial artifact stays libmpi-free.
    // `statuses` must be under-aligned in BOTH families or core.zig's
    // family-invariant `?[*]align(1) abi.Status` parameter stops matching.
    try std.testing.expectEqual(
        ?[*]align(1) mpich.Status,
        @typeInfo(@TypeOf(mpich.statusesIgnore)).@"fn".return_type.?,
    );
    try std.testing.expectEqual(
        ?[*]align(1) ompi.Status,
        @typeInfo(@TypeOf(ompi.statusesIgnore)).@"fn".return_type.?,
    );
    try std.testing.expectEqual(mpich.RawComm, @typeInfo(@TypeOf(mpich.commWorld)).@"fn".return_type.?);
    try std.testing.expectEqual(ompi.RawComm, @typeInfo(@TypeOf(ompi.commWorld)).@"fn".return_type.?);
}
