//! qmri — MRI quality factors of a PUFFY KDMP snapshot: how many grid
//! cells resolve the fastest-growing MRI wavelength λ_MRI = 2π v_A/Ω, per
//! direction, over the disk body.
//!
//! With the directional relativistic Alfvén speed v_A^(î) = |b^(î)|/√(ρh+b²)
//! the orthonormal factors √g_ii cancel between wavelength and cell size:
//!     Q_i = 2π |b^i| / ( √(ρh + b²) · |Ω| · Δx^i )      (coordinate b^i, Δx^i)
//! with Ω = u^φ/u^t. The inertia ρh includes the radiation enthalpy
//! (ρ + γu + (4/3)Ê) — the honest choice for a radiation-supported disk;
//! the gas-only variant is bigger by the reported √(w_tot/w_gas) factor.
//!
//! Statistics are mass-weighted over the DISK mask (σ = b²/ρ < 1,
//! ρ > 10³ × the atmosphere floor profile, r < 100) and sum-reducible on
//! purpose — the same shapes can later stream into scalars.dat under MPI.
//! Thresholds: Q_θ ≳ 10, Q_φ ≳ 20 (Sano+04; Hawley+11,13; EHT comparison).
//!
//! usage: qmri <params.toml> <file.kdmp>...

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const puffy = koral.problems.puffy;
const render = koral.render;
const metric = koral.metric.core;
const relele = koral.relele;
const mhd = koral.physics.mhd;
const radiation = koral.physics.radiation;
const L = koral.VarLayout(cfg);

const q_theta_min = 10.0;
const q_phi_min = 20.0;

fn analyze(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 32));
    defer allocator.free(bytes);
    var data = try render.DumpData.fromBytes(allocator, bytes);
    defer data.deinit(allocator);
    const h = data.header;
    if (h.nv != L.count) return error.DimMismatch;

    const g = puffy.makeGridNz(h.nx, h.ny, h.nz);
    const mp = puffy.mp;
    const gam = puffy.gam;

    var mass: f64 = 0; // Σ ρ·√-g over the disk mask
    var m_qr: f64 = 0;
    var m_qth: f64 = 0;
    var m_qph: f64 = 0;
    var m_qth_low: f64 = 0;
    var m_qph_low: f64 = 0;
    var m_unmag: f64 = 0;
    var m_beta: f64 = 0;
    var m_wratio: f64 = 0; // Σ (w_tot/w_gas)·ρ√-g
    var ncell: usize = 0;
    var mass_all: f64 = 0; // whole domain, for the mask fraction

    var iz: usize = 0;
    while (iz < h.nz) : (iz += 1) {
        var iy: usize = 0;
        while (iy < h.ny) : (iy += 1) {
            var ix: usize = 0;
            while (ix < h.nx) : (ix += 1) {
                const cell = data.cell(ix, iy, iz);
                var pp: [L.count]f64 = undefined;
                @memcpy(&pp, cell);
                const rho = pp[L.index(.rho)];
                const uu = pp[L.index(.uu)];
                if (!(rho > 0) or !(uu > 0)) continue;

                const x = [4]f64{ 0, g.xc(@intCast(ix)), g.yc(@intCast(iy)), g.zc(@intCast(iz)) };
                const r = @exp(x[1]) + mp.mksr0;
                const cd = metric.compute(.mks2, mp, x);
                const geom = render.geomFromCoordData(x, &cd);
                const wgt_all = rho * cd.gdet;
                mass_all += wgt_all;

                // disk mask: real material, not funnel, inflow-relevant radii
                if (r > 100.0) continue;
                const floor_rho = puffy.rhoatmmin * std.math.pow(f64, r / 2.0, -1.5);
                if (rho < 1.0e3 * floor_rho) continue;

                const u = relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    &geom,
                ) catch continue;
                const b = mhd.bconBcovBsqFrom4vel(
                    .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                    u.con,
                    u.cov,
                    &geom,
                );
                if (b.bsq > rho) continue; // σ-cut

                const omega = @abs(u.con[3] / u.con[0]);
                if (!(omega > 1e-12)) continue;

                // fluid-frame radiation energy for the total inertia
                var ehat: f64 = 0;
                const rij = radiation.calcRijG(cfg, f64, pp, geom.cov(), geom.con());
                for (0..4) |i| {
                    for (0..4) |j| ehat += rij[i][j] * u.cov[i] * u.cov[j];
                }
                if (!(ehat > 0) or !std.math.isFinite(ehat)) ehat = 0;

                const w_gas = rho + gam * uu;
                const w_tot = w_gas + (4.0 / 3.0) * ehat;
                const va_den = @sqrt(w_tot + b.bsq) * omega;

                const q_r = 2.0 * std.math.pi * @abs(b.bcon[1]) / (va_den * g.dx);
                const q_th = 2.0 * std.math.pi * @abs(b.bcon[2]) / (va_den * g.dy);
                const q_ph = 2.0 * std.math.pi * @abs(b.bcon[3]) / (va_den * g.dz);

                const wgt = wgt_all;
                mass += wgt;
                ncell += 1;
                m_qr += wgt * q_r;
                m_qth += wgt * q_th;
                m_qph += wgt * q_ph;
                if (q_th < q_theta_min) m_qth_low += wgt;
                if (q_ph < q_phi_min) m_qph_low += wgt;
                if (b.bsq < 1e-8 * rho) m_unmag += wgt; // essentially field-free
                const pmag = 0.5 * b.bsq;
                const ptot = (gam - 1.0) * uu + ehat / 3.0;
                if (pmag > 0) m_beta += wgt * @min(ptot / pmag, 1e6);
                m_wratio += wgt * (w_tot / w_gas);
            }
        }
    }

    if (mass <= 0) {
        std.debug.print("{s}: t={e:.4} — disk mask is empty\n", .{ path, h.t });
        return;
    }
    std.debug.print(
        "{s}\n  t={e:.4}  disk cells={d} ({d:.1}% of mass)  <w_tot/w_gas>={d:.1}\n" ++
            "  <Q_r>={d:.2}  <Q_th>={d:.2}  <Q_ph>={d:.2}   (mass-weighted, rad-inclusive inertia)\n" ++
            "  mass frac Q_th<{d}: {d:.1}%   Q_ph<{d}: {d:.1}%   field-free: {d:.1}%   <beta>={e:.2}\n",
        .{
            path,        h.t,                      ncell,                  100.0 * mass / mass_all, m_wratio / mass,
            m_qr / mass, m_qth / mass,             m_qph / mass,           q_theta_min,             100.0 * m_qth_low / mass,
            q_phi_min,   100.0 * m_qph_low / mass, 100.0 * m_unmag / mass, m_beta / mass,
        },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    const params_path = args.next() orelse {
        std.debug.print("usage: qmri <params.toml> <file.kdmp>...\n", .{});
        return error.BadArgs;
    };

    var p = koral.Params.load(allocator, io, params_path) catch |err| {
        std.debug.print("qmri: cannot load params '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);
    puffy.mass = p.mass;
    puffy.mp.a = p.bhspin;
    puffy.applyPhysicsOverrides(&p);
    puffy.rmin = if (p.rmin > 0.0) p.rmin else puffy.rminForSpin(p.bhspin);
    if (p.rmax > 0.0) puffy.rmax = p.rmax;

    var any = false;
    while (args.next()) |kpath| {
        any = true;
        analyze(allocator, io, kpath) catch |err| {
            std.debug.print("qmri: {s}: {s}\n", .{ kpath, @errorName(err) });
        };
    }
    if (!any) {
        std.debug.print("usage: qmri <params.toml> <file.kdmp>...\n", .{});
        return error.BadArgs;
    }
}
