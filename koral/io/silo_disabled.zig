//! Stub that stands in for the `silo` wrapper package when koral is built
//! without `-Dsilo`. It lets `@import("silo")` resolve (so the `koral` module
//! always compiles) without pulling in the from-source Silo C library or libc.
//!
//! `koral/io/silo.zig` references the real API only inside
//! `if (comptime build_options.silo)` branches, which are never analyzed when
//! Silo is off — so none of the members below are ever touched. They exist only
//! to make the namespace resolvable.

/// This build does NOT link Silo. (The real wrapper sets this to `true`.)
pub const available = false;

/// Placeholder for the translated C API. Never referenced when disabled.
pub const c = struct {};
