#include "voronoi_defs.h"

#include <assert.h>

#include <cmath>    // std::fabs, std::isfinite  (MSD_RPD_DEGEN_FILTER)
#include <cstdlib>  // std::getenv

void ConvexCellHost::print_info() const {
  printf(
      "Cell of id: %d, voro_id: %d, sq_radius: %f, tet_id: %d, nb_v: "
      "%d, nb_p: %d, nb_e: %d, status: %d, thread_id: %d, is_vertex_null: %d, "
      "euler: %f\n",
      id, voro_id, weight, tet_id, nb_v, nb_p, nb_e, status, thread_id,
      is_vertex_null, euler);

  FOR(i, nb_p) {
    // printf("clip %d, #adj %f, seed1/fid1: %f, seed2/fid2: %f\n", i,
    // clip(i).h,
    //        clip(i).g, clip(i).k);
    printf("clip %d, a: %f, b: %f c: %f, d: %f, clip_id2: (%d,%d)\n", i,
           clip_trans_const(i).x, clip_trans_const(i).y, clip_trans_const(i).z,
           clip_trans_const(i).w, clip_id2_const(i).x, clip_id2_const(i).y);
  }

  FOR(i, nb_v) {
    printf("vertex %d: (%d, %d, %d), #adj %d\n", i, ver_trans_const(i).x,
           ver_trans_const(i).y, ver_trans_const(i).z, ver_trans_const(i).w);
  }

  FOR(i, nb_e) {
    printf("edge %d: (%d, %d)\n", i, edge_id2_const(i).x, edge_id2_const(i).y);
  }
}

// Compensated (error-free-transformation) fp32 Cramer solve: same float
// storage and float arithmetic as `compute_vertex_coordinates`, but every
// product and sum carries its rounding error in a second float, so the 3x3
// determinants are evaluated to ~2x fp32 precision. No double anywhere.
namespace {
struct FF {
  float hi, lo;
};
inline FF ff(float a) { return FF{a, 0.f}; }
inline FF ff_add(FF a, FF b) {
  const float s = a.hi + b.hi;
  const float bb = s - a.hi;
  float err = (a.hi - (s - bb)) + (b.hi - bb);
  err += a.lo + b.lo;
  const float hi = s + err;
  return FF{hi, err - (hi - s)};
}
inline FF ff_neg(FF a) { return FF{-a.hi, -a.lo}; }
inline FF ff_mul(FF a, FF b) {
  const float p = a.hi * b.hi;
  float err = std::fmaf(a.hi, b.hi, -p);
  err += a.hi * b.lo + a.lo * b.hi;
  const float hi = p + err;
  return FF{hi, err - (hi - p)};
}
inline FF ff_det2(FF a, FF b, FF c, FF d) {
  return ff_add(ff_mul(a, d), ff_neg(ff_mul(b, c)));
}
inline FF ff_det3(float a11, float a12, float a13, float a21, float a22,
                  float a23, float a31, float a32, float a33) {
  const FF t1 = ff_mul(ff(a11), ff_det2(ff(a22), ff(a23), ff(a32), ff(a33)));
  const FF t2 = ff_mul(ff(a21), ff_det2(ff(a12), ff(a13), ff(a32), ff(a33)));
  const FF t3 = ff_mul(ff(a31), ff_det2(ff(a12), ff(a13), ff(a22), ff(a23)));
  return ff_add(ff_add(t1, ff_neg(t2)), t3);
}
// FF / FF to fp32 accuracy: one Newton correction on the fp32 quotient.
inline float ff_div(FF a, FF b) {
  const float q = a.hi / b.hi;
  const FF r = ff_add(a, ff_neg(ff_mul(ff(q), b)));
  return q + r.hi / b.hi;
}
}  // namespace

// ninwang: must in sync with ConvexCell::compute_vertex_coordinates()
cfloat4 ConvexCellHost::compute_vertex_coordinates(cuchar3 t,
                                                   bool persp_divide) const {
  // MSD_RPD_FP32_COMP (default ON) -- evaluate the Cramer solve with
  // compensated fp32 (ff_det3/ff_div above). MEASURED 2026-09-17 on
  // 15_tray_thin against an fp64 shadow of the SAME fp32 plane data, over the
  // 195603 vertices that pass degenerate_plane_triple: the plain expression
  // below leaves 487 vertices worse than 1e-6 relative, 174 worse than 1e-4 and
  // 34 worse than 1e-2, the worst 9.6% off; compensated the worst is 5.85e-08,
  // ~0.5 ulp. The vertex COUNT is identical either way -- the degeneracy filter
  // is untouched and still runs in plain fp32, so no topology decision moves.
  // `=0` restores the original expression exactly.
  static const bool comp = [] {
    const char* v = std::getenv("MSD_RPD_FP32_COMP");
    return !v || !*v || !(v[0] == '0');
  }();

  cfloat4 pi1 = clip_trans_const(t.x);
  cfloat4 pi2 = clip_trans_const(t.y);
  cfloat4 pi3 = clip_trans_const(t.z);

  if (comp) {
    const FF fx = ff_det3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w,
                          pi3.y, pi3.z);
    const FF fy = ff_det3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x,
                          pi3.w, pi3.z);
    const FF fz = ff_det3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x,
                          pi3.y, pi3.w);
    const FF fw = ff_det3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x,
                          pi3.y, pi3.z);
    if (persp_divide)
      return cmake_float4(-ff_div(fx, fw), -ff_div(fy, fw), -ff_div(fz, fw), 1);
    return cmake_float4(-fx.hi, -fy.hi, -fz.hi, fw.hi);
  }

  cfloat4 result;
  result.x =
      -cdet3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
  result.y =
      -cdet3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
  result.z =
      -cdet3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
  result.w =
      cdet3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);

  cfloat4 result_per = cmake_float4(result.x / result.w, result.y / result.w,
                                    result.z / result.w, 1);

  if (persp_divide) return result_per;
  return result;
}

void ConvexCellHost::reload_active() {
  // reload active clipping planes
  active_clipping_planes.clear();
  active_clipping_planes.resize(nb_p + 1, 0);
  std::map<int, std::set<int>> plane2v;
  FOR(t, nb_v) {
    active_clipping_planes[ver_trans(t).x]++;
    active_clipping_planes[ver_trans(t).y]++;
    active_clipping_planes[ver_trans(t).z]++;
    plane2v[ver_trans(t).x].insert(t);
    plane2v[ver_trans(t).y].insert(t);
    plane2v[ver_trans(t).z].insert(t);
  }

  // reload active edges
  active_edges.clear();
  active_edges.resize(nb_e + 1, 0);
  std::set<int> v_insect;
  FOR(eid, nb_e) {
    cuchar3 e = edge(eid);
    assert(e.x < _MAX_P_ && e.y < _MAX_P_);
    if (active_clipping_planes[e.x] <= 0 || active_clipping_planes[e.y] <= 0)
      continue;
    set_intersection(plane2v.at(e.x), plane2v.at(e.y), v_insect);
    if (v_insect.size() < 2) continue;  // edge not exist
    active_edges[eid]++;
  }

  // update the flag
  is_active_updated = true;
}

// MSD_RPD_DEGEN_FILTER (default ON) -- static filtered predicate for "are these
// three clipping planes a PENCIL", i.e. is the perspective divide in
// compute_vertex_coordinates a divide by zero.
//
// MEASURED 2026-09-16 on 15_tray_thin (site 2452 / tet 34235, dual triangle
// (6,5,4)): the three plane normals were
//     n4 = ( 7.618042, -20.842773,  2.476908)
//     n5 = ( 0.000000, -20.842773,  0.000000)
//     n6 = ( 7.618042,   0.000000,  2.476908)
// with n4 = n5 + n6 EXACTLY, so det = 0 exactly (fp32 0.0, rational 0, fp64
// -7e-14 of pure rounding noise). Three planes in a pencil have no finite
// common point: the homogeneous vertex is at infinity and the divide yields
// (-nan, -nan, -inf). The normals are centre-difference vectors, so this says
// the four sphere centres are COPLANAR -- generic on a plate, where the medial
// spheres lie on a sheet, which is why one thin tray produced ~40k of these.
//
// ⚠ MORE PRECISION MAKES THIS WORSE. In fp32 the determinant underflows to
// exactly 0.0 and the isnan guard fires. In fp64 it is -7e-14, the divide
// yields a FINITE coordinate of magnitude ~1e14, and the guard never sees it.
// So the test has to be a relative-magnitude filter, not a precision upgrade.
//
// The bound is the standard static filter: |det| <= eps * (sum of the absolute
// values of its six product terms) means the sign is not decidable in fp32.


static bool degenerate_plane_triple(const cfloat4& p1, const cfloat4& p2,
                                    const cfloat4& p3, float& det_out) {
  const float d = cdet3x3(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
  det_out = d;
  const float scale = std::fabs(p1.x * p2.y * p3.z) + std::fabs(p1.x * p2.z * p3.y) +
                      std::fabs(p1.y * p2.x * p3.z) + std::fabs(p1.y * p2.z * p3.x) +
                      std::fabs(p1.z * p2.x * p3.y) + std::fabs(p1.z * p2.y * p3.x);
  // 8 ulp of fp32 against the term magnitude; at scale == 0 every term is zero
  // and the triple is degenerate by definition.
  return !(std::fabs(d) > 8.0f * 1.1920929e-7f * scale);
}

void ConvexCellHost::reload_pc_explicit() {
  static const bool degen_filter = [] {
    const char* v = std::getenv("MSD_RPD_DEGEN_FILTER");
    return !v || !*v || !(v[0] == '0');
  }();
  static const bool degen_verbose = [] {
    const char* v = std::getenv("MSD_RPD_DEGEN_VERBOSE");
    return v && *v && v[0] != '0';
  }();
  // MSD_RPD_DEGEN_REPAIR (default ON) -- place a degenerate vertex ON its three
  // planes instead of at the origin, so the cell stays usable.
  //
  // A plane pencil means the four sphere centres are coplanar, so the three
  // bisectors meet along a LINE, not a point: the 3x3 system is rank 2 and
  // consistent, and any point on that line satisfies all three planes. The old
  // code pushed (0,0,0) and set `is_vertex_null`, and rpd_update.cxx then
  // dropped the ENTIRE convex cell -- every facet of it -- over one such
  // vertex. On a plate the coplanar quadruple is GENERIC (the medial spheres
  // lie on a sheet), so this discards power-cell topology wholesale.
  //
  // The repair solves the same three planes in a Tikhonov least-norm sense
  // anchored at the centroid of the cell's FINITE vertices: it returns the
  // point on the pencil line nearest that centroid, reduces to the exact vertex
  // when the triple is well conditioned, and stays finite when the planes are
  // inconsistent (parallel) rather than merely dependent. `n_degenerate_v`
  // still counts every repair, so the census is unchanged.
  // `=0` restores the placeholder + whole-cell drop.
  static const bool degen_repair = [] {
    const char* v = std::getenv("MSD_RPD_DEGEN_REPAIR");
    return !v || !*v || !(v[0] == '0');
  }();

  pc_points.clear();
  n_degenerate_v = 0;
  std::vector<int> degen_lv;  // local vertex ids needing a repair
  FOR(i, nb_v) {
    const cuchar3 t = cmake_uchar3(ver_trans(i));
    if (degen_filter) {
      float det = 0.f;
      if (degenerate_plane_triple(clip_trans_const(t.x), clip_trans_const(t.y),
                                  clip_trans_const(t.z), det)) {
        // The vertex is at infinity. Mark the cell, COUNT it, and push a
        // placeholder so `pc_points` stays index-parallel with `nb_v` -- the
        // old code `return`ed here, leaving a SHORT array, and every consumer
        // that indexes pc_points by vertex id then reads the wrong point or
        // runs off the end (the [FixFacet] map::at abort).
        ++n_degenerate_v;
        if (degen_repair)
          degen_lv.push_back(i);
        else
          is_vertex_null = true;
        if (degen_verbose)
          printf("DEGEN: site %d tet %d dual triangle (%d,%d,%d) det %.6g "
                 "-- plane pencil, vertex at infinity\n",
                 voro_id, tet_id, t.x, t.y, t.z, det);
        pc_points.push_back(cmake_float3(0.f, 0.f, 0.f));
        continue;
      }
    }
    cfloat4 voro_vertex = compute_vertex_coordinates(t);
    // Non-finite covers inf as well as NaN: with the filter off, or if a
    // near-degenerate triple slips the bound, the divide can produce a finite
    // -inf that `isnan` alone lets through.
    if (!std::isfinite(voro_vertex.x) || !std::isfinite(voro_vertex.y) ||
        !std::isfinite(voro_vertex.z)) {
      ++n_degenerate_v;
      if (degen_repair)
        degen_lv.push_back(i);
      else
        is_vertex_null = true;
      if (degen_verbose)
        printf("ERROR: cell of site %d tet %d has vertex NON-FINITE (%f, %f, "
               "%f), dual triangle (%d, %d, %d)\n",
               voro_id, tet_id, voro_vertex.x, voro_vertex.y, voro_vertex.z,
               t.x, t.y, t.z);
      pc_points.push_back(cmake_float3(0.f, 0.f, 0.f));
      continue;
    }
    pc_points.push_back(
        cmake_float3(voro_vertex.x, voro_vertex.y, voro_vertex.z));
  }

  if (!degen_lv.empty()) {
    // Anchor: centroid of the finite vertices. A cell with NO finite vertex has
    // nothing to anchor to, so it keeps the old whole-cell drop.
    const int n_fin = (int)nb_v - (int)degen_lv.size();
    if (n_fin <= 0) {
      is_vertex_null = true;
    } else {
      float cx = 0.f, cy = 0.f, cz = 0.f;
      {
        size_t k = 0;
        FOR(i, nb_v) {
          if (k < degen_lv.size() && degen_lv[k] == (int)i) {
            ++k;
            continue;
          }
          cx += pc_points[i].x;
          cy += pc_points[i].y;
          cz += pc_points[i].z;
        }
      }
      cx /= (float)n_fin;
      cy /= (float)n_fin;
      cz /= (float)n_fin;
      for (int lv : degen_lv) {
        const cuchar3 t = cmake_uchar3(ver_trans(lv));
        const cfloat4 p[3] = {clip_trans_const(t.x), clip_trans_const(t.y),
                              clip_trans_const(t.z)};
        // Pick the best-conditioned PAIR of the three planes: their normals'
        // cross product is the direction of the pencil line L. The third plane
        // is dependent, so its constraint is implied and dropping it costs
        // nothing -- and when the triple is inconsistent rather than merely
        // dependent, the two best-conditioned planes are still the right answer.
        int ba = -1, bb = -1;
        float ux = 0.f, uy = 0.f, uz = 0.f, best = -1.f;
        for (int a = 0; a < 3; ++a) {
          const int b = (a + 1) % 3;
          const float vx = p[a].y * p[b].z - p[a].z * p[b].y;
          const float vy = p[a].z * p[b].x - p[a].x * p[b].z;
          const float vz = p[a].x * p[b].y - p[a].y * p[b].x;
          const float m2 = vx * vx + vy * vy + vz * vz;
          if (m2 > best) {
            best = m2;
            ba = a;
            bb = b;
            ux = vx;
            uy = vy;
            uz = vz;
          }
        }
        // All three normals parallel: no line, nothing to place the vertex on.
        if (!(best > 0.f)) {
          is_vertex_null = true;
          continue;
        }
        // Rows (n_a, n_b, u) are mutually independent by construction -- u is
        // orthogonal to both normals -- so this 3x3 is well conditioned even
        // though the original triple was not. The third row pins the position
        // ALONG L to the projection of the anchor, giving the point on L
        // nearest the cell's finite centroid.
        const float rhs0 = -p[ba].w, rhs1 = -p[bb].w;
        const float rhs2 = ux * cx + uy * cy + uz * cz;
        const FF dw = ff_det3(p[ba].x, p[ba].y, p[ba].z, p[bb].x, p[bb].y,
                              p[bb].z, ux, uy, uz);
        const FF dx = ff_det3(rhs0, p[ba].y, p[ba].z, rhs1, p[bb].y, p[bb].z,
                              rhs2, uy, uz);
        const FF dy = ff_det3(p[ba].x, rhs0, p[ba].z, p[bb].x, rhs1, p[bb].z,
                              ux, rhs2, uz);
        const FF dz = ff_det3(p[ba].x, p[ba].y, rhs0, p[bb].x, p[bb].y, rhs1,
                              ux, uy, rhs2);
        const float x = ff_div(dx, dw);
        const float y = ff_div(dy, dw);
        const float z = ff_div(dz, dw);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          is_vertex_null = true;
          continue;
        }
        pc_points[lv] = cmake_float3(x, y, z);
      }
    }
  }

  // some clipping planes may not exist in tri but
  // we still sotre it, here is to filter those planes
  if (!is_active_updated) reload_active();
  assert(is_active_updated);

  pc_local_active_faces.clear();
  pc_lf2active_map.clear();
  pc_lf2active_map.resize(nb_p, -1);
  int lf = 0;  // local active fid

  FOR(plane, nb_p) {
    if (active_clipping_planes[plane] > 0) {
      std::vector<int> tab_v;   // index of dual vertex
      std::vector<int> tab_lp;  // local index of dual vertex in dual triangle
      // for each dual triangle
      FOR(t, nb_v) {
        // store info of dual vertex
        if ((int)ver_trans(t).x == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(0);
        } else if ((int)ver_trans(t).y == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(1);
        } else if ((int)ver_trans(t).z == plane) {
          tab_v.push_back(t);
          tab_lp.push_back(2);
        }
      }

      if (tab_lp.size() <= 2) {
        std::cout << (int)plane << std::endl;
      }

      int i = 0;
      int j = 0;
      pc_local_active_faces.push_back(std::vector<int>(0));

      while (pc_local_active_faces[lf].size() < tab_lp.size()) {
        int ind_i = (tab_lp[i] + 1) % 3;
        bool temp = false;
        j = 0;
        while (temp == false) {
          int ind_j = (tab_lp[j] + 2) % 3;
          if ((int)ith_plane(tab_v[i], ind_i) ==
              (int)ith_plane(tab_v[j], ind_j)) {
            pc_local_active_faces[lf].push_back(tab_v[i]);
            temp = true;
            i = j;
          }
          j++;
        }
      }
      pc_lf2active_map[plane] = lf;  // map from lf -> laf (active)
      lf++;
    }
  }
  is_pc_explicit = true;
}

Scalar ConvexCellHost::cal_cell_euler() {
  Scalar sum_v = 0.f, sum_f = 0.f, sum_e = 0.f;

  if (!is_active_updated) reload_active();
  assert(is_active_updated);

  // vertices
  FOR(vid, nb_v) {
    int v_adj = static_cast<int>(ver_trans(vid).w);
    assert(v_adj > 0);
    sum_v += 1.f / v_adj;
  }

  // faces
  FOR(fid, nb_p) {
    assert(fid < _MAX_P_);
    if (active_clipping_planes[fid] <= 0) continue;
    assert(clip_trans(fid).h == F_CELL_ADJ_DEFAULT ||
           clip_trans(fid).h == F_TET_ADJ_DEFAULT);
    sum_f += 1. / clip_trans(fid).h;
  }

  // edges
  FOR(eid, nb_e) {
    cuchar3 e = edge(eid);
    assert(e.x < _MAX_P_ && e.y < _MAX_P_);
    if (active_edges[eid] <= 0) continue;
    sum_e += 1. / e.z;
  }

  euler = sum_v - sum_e + sum_f;
  return euler;
}

Scalar ConvexCellHost::cal_halfplane_facet_euler(int neigh_id) {
  Scalar sum_v = 0.f, sum_e = 0.f;

  // find the clip_id on halfplane defined by [neigh_id, seed_id]
  int clip_id = -1;
  if (!is_active_updated) reload_active();
  assert(is_active_updated);
  FOR(fid, nb_p) {
    assert(fid < _MAX_P_);
    if (active_clipping_planes[fid] <= 0) continue;
    assert(clip_trans(fid).h == F_CELL_ADJ_DEFAULT ||
           clip_trans(fid).h == F_TET_ADJ_DEFAULT);
    cint2 f_id2 = clip_id2_const(fid);
    if (f_id2.x == neigh_id || f_id2.y == neigh_id) {
      clip_id = fid;
      break;
    }
  }
  // face must on halfplane
  assert(clip_id != -1);
  assert(clip_trans(clip_id).h == F_CELL_ADJ_DEFAULT);
  Scalar sum_f = 1.f;

  // vertices on face clip_id
  FOR(vid, nb_v) {
    cuchar4 v = ver_trans(vid);
    if (v.x != clip_id && v.y != clip_id && v.z != clip_id) continue;
    int v_adj = static_cast<int>(v.w);
    assert(v_adj > 0);
    sum_v += 1.f / v_adj;
  }

  // edges on face clip_id
  FOR(eid, nb_e) {
    cuchar3 e = edge(eid);
    assert(e.x < _MAX_P_ && e.y < _MAX_P_);
    if (e.x != clip_id && e.y != clip_id) continue;
    if (active_edges[eid] <= 0) continue;
    sum_e += 1. / e.z;
  }

  float facet_euler = sum_v - sum_e + sum_f;
  return facet_euler;
}

void ConvexCellHost::print_cell_detail_euler() const {
  printf("--------- \n");
  assert(is_active_updated);
  FOR(v, nb_v) {
    printf("[Detail] seed %d id %d has vertex %d with #adj_cells: %d \n",
           voro_id, id, v, ver_data_trans[v].w);
  }

  FOR(e, nb_e) {
    if (active_edges[e] <= 0) continue;
    printf("[Detail] seed %d id %d has edge %d: (%d, %d) with #adj_cells: %d\n",
           voro_id, id, e, edge_data[e].x, edge_data[e].y, edge_data[e].z);
  }

  FOR(f, nb_p) {
    if (active_clipping_planes[f] <= 0) continue;
    cint2 hp = clip_id2_const(f);
    printf("[Detail] seed %d id %d has plane %d with #adj_cells: %f \n",
           voro_id, id, f, clip_data_trans[f].h);
    // printf("[Detail] seed %d id %d has plane %d with hp.x: %d, hp.y: %d \n",
    //        voro_id, id, f, hp.x, hp.y);
  }
}

bool is_convex_cell_valid(const ConvexCellHost& cell) {
  if (cell.status != Status::success &&
      cell.status != Status::security_radius_not_reached)
    return false;
  return true;
}
