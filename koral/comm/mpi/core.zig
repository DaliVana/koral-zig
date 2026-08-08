//! The hand-written extern MPI core (MPI plan §3.2): the small function
//! subset koral needs, declared over the comptime-selected family's raw
//! handle types. Signatures are family-invariant given those types — this
//! file never changes when a family is added. No `@cImport`/translate-c
//! anywhere (0.16's Aro chokes on OMPI's link-time constants and JIT-
//! compiles for minutes; ~20 externs are decisively cheaper).

const abi = @import("abi.zig");

pub extern fn MPI_Init_thread(argc: ?*c_int, argv: ?*anyopaque, required: c_int, provided: *c_int) c_int;
pub extern fn MPI_Initialized(flag: *c_int) c_int;
pub extern fn MPI_Finalize() c_int;
pub extern fn MPI_Abort(comm: abi.RawComm, errorcode: c_int) c_int;

pub extern fn MPI_Comm_rank(comm: abi.RawComm, rank: *c_int) c_int;
pub extern fn MPI_Comm_size(comm: abi.RawComm, size: *c_int) c_int;
pub extern fn MPI_Comm_free(comm: *abi.RawComm) c_int;

pub extern fn MPI_Cart_create(old: abi.RawComm, ndims: c_int, dims: [*]const c_int, periods: [*]const c_int, reorder: c_int, cart: *abi.RawComm) c_int;
pub extern fn MPI_Cart_shift(comm: abi.RawComm, direction: c_int, disp: c_int, rank_source: *c_int, rank_dest: *c_int) c_int;

pub extern fn MPI_Send_init(buf: ?*const anyopaque, count: c_int, datatype: abi.RawDatatype, dest: c_int, tag: c_int, comm: abi.RawComm, request: *abi.RawRequest) c_int;
pub extern fn MPI_Recv_init(buf: ?*anyopaque, count: c_int, datatype: abi.RawDatatype, source: c_int, tag: c_int, comm: abi.RawComm, request: *abi.RawRequest) c_int;
pub extern fn MPI_Startall(count: c_int, requests: [*]abi.RawRequest) c_int;
// `statuses` is under-aligned on purpose: MPICH's MPI_STATUSES_IGNORE is the
// address 1, which no aligned pointer type can hold (see abi_mpich.zig).
pub extern fn MPI_Waitall(count: c_int, requests: [*]abi.RawRequest, statuses: ?[*]align(1) abi.Status) c_int;
pub extern fn MPI_Request_free(request: *abi.RawRequest) c_int;

pub extern fn MPI_Allreduce(sendbuf: ?*const anyopaque, recvbuf: ?*anyopaque, count: c_int, datatype: abi.RawDatatype, op: abi.RawOp, comm: abi.RawComm) c_int;
pub extern fn MPI_Barrier(comm: abi.RawComm) c_int;
pub extern fn MPI_Wtime() f64;

// MPI-IO (plan §8.1: the collective KDMP path — write_at_all at closed-form
// offsets; no file views, no subarray types). The single-status pointer is
// `align(1)` for the same reason MPI_Waitall's is: MPICH's STATUS_IGNORE
// sentinel is the address 1.
pub extern fn MPI_File_open(comm: abi.RawComm, filename: [*:0]const u8, amode: c_int, info: abi.RawInfo, fh: *abi.RawFile) c_int;
pub extern fn MPI_File_close(fh: *abi.RawFile) c_int;
pub extern fn MPI_File_set_size(fh: abi.RawFile, size: abi.Offset) c_int;
pub extern fn MPI_File_get_size(fh: abi.RawFile, size: *abi.Offset) c_int;
pub extern fn MPI_File_write_at(fh: abi.RawFile, offset: abi.Offset, buf: ?*const anyopaque, count: c_int, datatype: abi.RawDatatype, status: ?*align(1) abi.Status) c_int;
pub extern fn MPI_File_write_at_all(fh: abi.RawFile, offset: abi.Offset, buf: ?*const anyopaque, count: c_int, datatype: abi.RawDatatype, status: ?*align(1) abi.Status) c_int;
pub extern fn MPI_File_read_at_all(fh: abi.RawFile, offset: abi.Offset, buf: ?*anyopaque, count: c_int, datatype: abi.RawDatatype, status: ?*align(1) abi.Status) c_int;
