//! The MPI communication backend (MPI plan §5-§7): a 1D periodic φ-ring of
//! NTZ ranks (`MPI_Cart_create`) and a single-stage zero-copy halo exchange:
//! four persistent channels (`Send_init`/`Recv_init`) bound directly into
//! the primitives Field's contiguous z-slabs. Each exchange is exactly
//! `Startall` + `Waitall`. The backend also provides in-place f64 collectives.
//! Thread level is
//! FUNNELED: every call here happens on the main thread between team
//! regions (threading.zig's existing contract).
//!
//! `comm/serial.zig` mirrors this API as no-ops; `comm/comm.zig` selects
//! between them at comptime on build_options.mpi.

const std = @import("std");
const abi = @import("abi.zig");
const core = @import("core.zig");
const Grid = @import("../../grid.zig").Grid;

/// Message tags: planes moving toward −φ (my lo domain slab → zlo neighbor's
/// hi ghosts) and toward +φ. Two distinct tags keep the ntz==2 case (both
/// neighbors are the same rank) unambiguous.
const tag_down: c_int = 100;
const tag_up: c_int = 101;

/// Any MPI failure mid-run is unrecoverable for a synchronous stencil code:
/// print and MPI_Abort (mirrors every surveyed production code).
fn check(rc: c_int, what: []const u8) void {
    if (rc != abi.success) {
        std.debug.print("koral/mpi: {s} failed (rc={d}) — aborting\n", .{ what, rc });
        _ = core.MPI_Abort(abi.commWorld(), 1);
        // MPI_Abort is specified not to return; `unreachable` here would be
        // UB in ReleaseFast if it ever did. Same belt-and-braces as abortJob.
        std.process.exit(1);
    }
}

pub const Mpi = struct {
    pub const enabled = true;

    /// The φ-ring communicator (with φ-only decomposition the world IS the
    /// ring: no sub-communicators exist anywhere, plan §7.2).
    cart: abi.RawComm,
    tk: usize,
    ntz: usize,
    /// Ring neighbors (cart ranks): toward −φ and +φ.
    nbr_lo: c_int,
    nbr_hi: c_int,
    /// Persistent exchange channels: {recv lo, recv hi, send lo, send hi}.
    /// Receives first so Startall posts them before the sends are started
    /// (C KORAL's send-before-recv anti-pattern is unexpressible here).
    reqs: [4]abi.RawRequest = @splat(abi.request_null),
    n_reqs: usize = 0,

    /// MPI_Init_thread(FUNNELED): call once, before Team spawn, before any
    /// other backend use. Idempotent (guards on MPI_Initialized).
    pub fn initWorld() !void {
        var inited: c_int = 0;
        check(core.MPI_Initialized(&inited), "MPI_Initialized");
        if (inited != 0) return;
        var provided: c_int = 0;
        check(core.MPI_Init_thread(null, null, abi.thread_funneled, &provided), "MPI_Init_thread");
        if (provided < abi.thread_funneled) return error.MpiThreadLevelTooLow;
    }

    /// Tear the job down before a communicator exists; the startup window
    /// between `initWorld` and the driver's `errdefer abortJob`. A
    /// rank-local failure there (an unreadable params file or opacity
    /// table on this node, a bad allocation) would otherwise return
    /// through `defer finalizeWorld`, and `MPI_Finalize` is a job-wide
    /// fence: the peers, already inside `MPI_Cart_create`, would wait out
    /// the entire wall-clock allocation.
    pub fn abortWorld(code: u8) noreturn {
        _ = core.MPI_Abort(abi.commWorld(), code);
        std.process.exit(code);
    }

    pub fn finalizeWorld() void {
        var inited: c_int = 0;
        _ = core.MPI_Initialized(&inited);
        if (inited != 0) _ = core.MPI_Finalize();
    }

    pub fn worldSize() usize {
        var n: c_int = 0;
        check(core.MPI_Comm_size(abi.commWorld(), &n), "MPI_Comm_size");
        return @intCast(n);
    }

    /// Build the periodic φ-ring. `ntz` must equal the world size (1 rank
    /// per node × node count; the launch line and params must agree).
    pub fn init(ntz: usize) !Mpi {
        if (worldSize() != ntz) return error.RankCountMismatch;
        var dims = [1]c_int{@intCast(ntz)};
        var periods = [1]c_int{1};
        var cart: abi.RawComm = undefined;
        check(core.MPI_Cart_create(abi.commWorld(), 1, &dims, &periods, 1, &cart), "MPI_Cart_create");
        var crank: c_int = 0;
        check(core.MPI_Comm_rank(cart, &crank), "MPI_Comm_rank");
        var lo: c_int = 0;
        var hi: c_int = 0;
        check(core.MPI_Cart_shift(cart, 0, 1, &lo, &hi), "MPI_Cart_shift");
        return .{ .cart = cart, .tk = @intCast(crank), .ntz = ntz, .nbr_lo = lo, .nbr_hi = hi };
    }

    pub fn deinit(self: *Mpi) void {
        self.unbindExchange();
        _ = core.MPI_Comm_free(&self.cart);
        self.* = undefined;
    }

    pub fn rank(self: *const Mpi) usize {
        return self.tk;
    }
    pub fn size(self: *const Mpi) usize {
        return self.ntz;
    }

    /// Bind the four persistent zero-copy channels into a Field's storage
    /// (plan §6.2). `data` is the AoS array (iv fastest, z slowest), `nv`
    /// its per-cell variable count. Each slab is NG full ghost-extended
    /// SX·SY planes; contiguous by layout, so receives land straight in
    /// the ghost planes and sends read the live domain edge planes. The
    /// storage must never move afterwards (Field.data never does).
    pub fn bindExchange(self: *Mpi, data: []f64, g: Grid, nv: usize) !void {
        std.debug.assert(self.n_reqs == 0); // bind once per Sim
        if (self.ntz == 1) return; // periodic self-wrap stays a BC concern
        const ng = g.ngz;
        std.debug.assert(ng > 0 and g.nz >= ng);
        const plane = g.sx() * g.sy() * nv; // one z-plane, ghosts included
        const slab = ng * plane;
        if (slab > std.math.maxInt(c_int)) return error.SlabTooLarge;
        const n: c_int = @intCast(slab);
        std.debug.assert(data.len == g.cellCount() * nv);

        const recv_lo = data[0..slab]; // ghost planes iz ∈ [-ng, 0)
        const send_lo = data[ng * plane ..][0..slab]; // domain iz ∈ [0, ng)
        const send_hi = data[g.nz * plane ..][0..slab]; // domain iz ∈ [nz-ng, nz)
        const recv_hi = data[(g.nz + ng) * plane ..][0..slab]; // ghosts iz ∈ [nz, nz+ng)

        const dt = abi.dtDouble();
        check(core.MPI_Recv_init(recv_lo.ptr, n, dt, self.nbr_lo, tag_up, self.cart, &self.reqs[0]), "MPI_Recv_init(lo)");
        check(core.MPI_Recv_init(recv_hi.ptr, n, dt, self.nbr_hi, tag_down, self.cart, &self.reqs[1]), "MPI_Recv_init(hi)");
        check(core.MPI_Send_init(send_lo.ptr, n, dt, self.nbr_lo, tag_down, self.cart, &self.reqs[2]), "MPI_Send_init(lo)");
        check(core.MPI_Send_init(send_hi.ptr, n, dt, self.nbr_hi, tag_up, self.cart, &self.reqs[3]), "MPI_Send_init(hi)");
        self.n_reqs = 4;
    }

    pub fn unbindExchange(self: *Mpi) void {
        for (self.reqs[0..self.n_reqs]) |*r| _ = core.MPI_Request_free(r);
        self.n_reqs = 0;
    }

    /// One exchange episode. No pack, no unpack, no buffers, no team.
    pub fn exchange(self: *Mpi) void {
        if (self.n_reqs == 0) return;
        check(core.MPI_Startall(@intCast(self.n_reqs), &self.reqs), "MPI_Startall");
        check(core.MPI_Waitall(@intCast(self.n_reqs), &self.reqs, abi.statusesIgnore()), "MPI_Waitall");
    }

    pub fn allreduceMax(self: *const Mpi, buf: []f64) void {
        self.allreduce(buf, abi.opMax());
    }
    pub fn allreduceMin(self: *const Mpi, buf: []f64) void {
        self.allreduce(buf, abi.opMin());
    }
    pub fn allreduceSum(self: *const Mpi, buf: []f64) void {
        self.allreduce(buf, abi.opSum());
    }

    fn allreduce(self: *const Mpi, buf: []f64, op: abi.RawOp) void {
        check(core.MPI_Allreduce(abi.inPlace(), buf.ptr, @intCast(buf.len), abi.dtDouble(), op, self.cart), "MPI_Allreduce");
    }

    pub fn barrier(self: *const Mpi) void {
        check(core.MPI_Barrier(self.cart), "MPI_Barrier");
    }

    // ---- MPI-IO (plan §8.1: collective KDMP checkpoints) ---------------

    /// An open MPI file handle. Opened/closed collectively on the ring.
    pub const File = struct { fh: abi.RawFile };

    /// Collectively create (or truncate) `path` at exactly `total` bytes.
    /// A failed open returns an error; it is uniform across ranks (the
    /// open is collective), so normal propagation stays coordinated.
    pub fn fileCreate(self: *const Mpi, path: [*:0]const u8, total: u64) !File {
        var fh: abi.RawFile = undefined;
        const rc = core.MPI_File_open(self.cart, path, abi.mode_create | abi.mode_wronly, abi.infoNull(), &fh);
        if (rc != abi.success) return error.MpiFileOpen;
        check(core.MPI_File_set_size(fh, @intCast(total)), "MPI_File_set_size");
        return .{ .fh = fh };
    }

    /// Collectively open `path` read-only (restart).
    pub fn fileOpenRead(self: *const Mpi, path: [*:0]const u8) !File {
        var fh: abi.RawFile = undefined;
        const rc = core.MPI_File_open(self.cart, path, abi.mode_rdonly, abi.infoNull(), &fh);
        if (rc != abi.success) return error.MpiFileOpen;
        return .{ .fh = fh };
    }

    pub fn fileClose(self: *const Mpi, f: *File) void {
        _ = self;
        check(core.MPI_File_close(&f.fh), "MPI_File_close");
    }

    /// Force written data out to storage. COLLECTIVE; every rank calls it.
    /// Used to order a checkpoint's body before its header (see the driver's
    /// writePrimDumpMpi): without that ordering the header can reach disk
    /// while the body it describes does not.
    pub fn fileSync(self: *const Mpi, f: *File) void {
        _ = self;
        check(core.MPI_File_sync(f.fh), "MPI_File_sync");
    }

    /// Size of an open file in bytes.
    ///
    /// This is how a short/corrupt checkpoint is detected, and the choice of
    /// *size* over a per-transfer status count is deliberate: every rank gets
    /// the same answer, so every rank reaches the same accept/reject decision
    /// and they stay collectively in step. A per-rank status count could have
    /// one rank error out of the sequence while its peers proceed into the
    /// next collective; trading a corrupt restart for a job-wide hang.
    pub fn fileSize(self: *const Mpi, f: *File) u64 {
        _ = self;
        var n: abi.Offset = 0;
        check(core.MPI_File_get_size(f.fh, &n), "MPI_File_get_size");
        return @intCast(n);
    }

    /// Pick count/datatype for a transfer: 8-byte multiples go as doubles
    /// (a whole KDMP body can exceed the 2 GiB c_int byte limit; /8 keeps
    /// every realistic slab in range), anything else as raw bytes.
    fn rwCount(len: usize) struct { count: c_int, dt: abi.RawDatatype } {
        // Hard checks, not asserts: these are compiled out in ReleaseFast —
        // exactly the build a production checkpoint runs under — leaving an
        // @intCast that is UB on overflow. A grid past the count limit must
        // fail loudly, not silently write the wrong number of elements.
        const n: usize = if (len % 8 == 0) len / 8 else len;
        if (n > std.math.maxInt(c_int)) {
            std.debug.print(
                "koral/mpi: transfer of {d} bytes exceeds the MPI element-count limit — aborting\n",
                .{len},
            );
            _ = core.MPI_Abort(abi.commWorld(), 1);
            std.process.exit(1);
        }
        return if (len % 8 == 0)
            .{ .count = @intCast(n), .dt = abi.dtDouble() }
        else
            .{ .count = @intCast(n), .dt = abi.dtByte() };
    }

    /// Collective write at a per-rank offset (every rank must call, each
    /// with its own slab). Failure mid-write is unrecoverable → abort.
    pub fn fileWriteAtAll(self: *const Mpi, f: *File, offset: u64, bytes: []const u8) void {
        _ = self;
        const c = rwCount(bytes.len);
        check(core.MPI_File_write_at_all(f.fh, @intCast(offset), bytes.ptr, c.count, c.dt, abi.statusIgnore()), "MPI_File_write_at_all");
    }

    /// Non-collective write (rank 0's 44-byte global header).
    pub fn fileWriteAt(self: *const Mpi, f: *File, offset: u64, bytes: []const u8) void {
        _ = self;
        const c = rwCount(bytes.len);
        check(core.MPI_File_write_at(f.fh, @intCast(offset), bytes.ptr, c.count, c.dt, abi.statusIgnore()), "MPI_File_write_at");
    }

    /// Collective read at a per-rank offset (identical offsets are legal ;
    /// the header read passes offset 0 on every rank).
    pub fn fileReadAtAll(self: *const Mpi, f: *File, offset: u64, bytes: []u8) void {
        _ = self;
        const c = rwCount(bytes.len);
        check(core.MPI_File_read_at_all(f.fh, @intCast(offset), bytes.ptr, c.count, c.dt, abi.statusIgnore()), "MPI_File_read_at_all");
    }

    /// Tear the whole job down NOW; never returns.
    ///
    /// This is the only safe way to leave a run from a RANK-LOCAL failure.
    /// Returning an error instead would unwind into the collective teardown
    /// (`MPI_Comm_free`, then `MPI_Finalize`'s job-wide fence) while the
    /// peers are still blocked in `MPI_Waitall`/`MPI_Allreduce`; nobody
    /// progresses and nobody exits, so a crash that should take seconds
    /// silently burns the entire wall-clock allocation.
    pub fn abortJob(self: *const Mpi, code: u8) noreturn {
        _ = self;
        _ = core.MPI_Abort(abi.commWorld(), code);
        // MPI_Abort is specified not to return; belt-and-braces if it does.
        std.process.exit(code);
    }
};
