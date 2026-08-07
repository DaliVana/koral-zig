//! The MPI communication backend (MPI plan §5–§7): a 1D periodic φ-ring of
//! NTZ ranks (`MPI_Cart_create`), a single-stage ZERO-COPY halo exchange —
//! four persistent channels (`Send_init`/`Recv_init`) bound directly into
//! the primitives Field's contiguous z-slabs, an episode being exactly
//! `Startall` + `Waitall` — and in-place f64 collectives. Thread level is
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
        unreachable;
    }
}

pub const Mpi = struct {
    pub const enabled = true;

    /// The φ-ring communicator (with φ-only decomposition the world IS the
    /// ring — no sub-communicators exist anywhere, plan §7.2).
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

    /// MPI_Init_thread(FUNNELED) — call once, before Team spawn, before any
    /// other backend use. Idempotent (guards on MPI_Initialized).
    pub fn initWorld() !void {
        var inited: c_int = 0;
        check(core.MPI_Initialized(&inited), "MPI_Initialized");
        if (inited != 0) return;
        var provided: c_int = 0;
        check(core.MPI_Init_thread(null, null, abi.thread_funneled, &provided), "MPI_Init_thread");
        if (provided < abi.thread_funneled) return error.MpiThreadLevelTooLow;
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
    /// SX·SY planes — contiguous by layout, so receives land straight in
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

    /// Tear the whole job down NOW; never returns.
    ///
    /// This is the only safe way to leave a run from a RANK-LOCAL failure.
    /// Returning an error instead would unwind into the collective teardown
    /// (`MPI_Comm_free`, then `MPI_Finalize`'s job-wide fence) while the
    /// peers are still blocked in `MPI_Waitall`/`MPI_Allreduce` — nobody
    /// progresses and nobody exits, so a crash that should take seconds
    /// silently burns the entire wall-clock allocation.
    pub fn abortJob(self: *const Mpi, code: u8) noreturn {
        _ = self;
        _ = core.MPI_Abort(abi.commWorld(), code);
        // MPI_Abort is specified not to return; belt-and-braces if it does.
        std.process.exit(code);
    }
};
