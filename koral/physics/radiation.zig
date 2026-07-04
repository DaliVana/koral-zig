//! M1 radiation basics. M4 needs only the lab-frame radiation stress
//! tensor for the advective flux; the rest of the M1 machinery (inversion,
//! wavespeeds with τ-limiting, four-force) arrives with M7/M8.

const relele = @import("../relele.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

const four_third: f64 = 4.0 / 3.0;
const one_third: f64 = 1.0 / 3.0;

/// Lab-frame R^μν of the M1 closure (C: calc_Rij_M1, rad.c:3098):
/// R^μν = (4/3) Ê u_r^μ u_r^ν + (1/3) Ê g^μν.
pub fn calcRij(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error![4][4]f64 {
    const L = layout.VarLayout(cfg);
    const erf = pp[L.index(.ee)];
    const urf = try relele.convVels(
        .{ 0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] },
        .velr,
        .vel4,
        &geom.gg,
        &geom.GG,
    );

    var rij: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            rij[i][j] = four_third * erf * urf[i] * urf[j] + one_third * erf * geom.GG[i][j];
        }
    }
    return rij;
}
