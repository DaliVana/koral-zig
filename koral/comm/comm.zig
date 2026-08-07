//! The comm seam (architecture doc §8, MPI plan §3.3): one comptime switch
//! selects the backend every consumer builds against. Serial builds never
//! analyze the MPI bindings (no libmpi anywhere); `-Dmpi` builds swap in
//! the real thing with an identical API.

pub const enabled = @import("build_options").mpi;

pub const Backend = if (enabled)
    @import("mpi/mpi.zig").Mpi
else
    @import("serial.zig").Serial;

pub const decomp = @import("decomp.zig");
pub const Decomp = decomp.Decomp;
pub const decompose = decomp.decompose;
