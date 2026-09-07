#pragma once

#include "input_types.h"
#include "medial_sphere.h"
#include "triangulation.h"
#include "voronoi_defs.h"

class RPD3D_GPU {
 public:
  RPD3D_GPU() {};
  ~RPD3D_GPU();
  void init(const TetMesh* _tet_mesh, const SurfaceMesh* _sf_mesh,
            const Parameter* _params);
  void set_spheres(std::vector<MedialSphere>* _all_medial_spheres);

 public:
  void calculate();
  void calculate_partial(int& num_itr_global, int& num_sphere_added,
                         const bool is_given_all_tets);

 public:
  void update_spheres_power_cells(bool is_compute_se_sfids = true);
  void load_partial_spheres_to_sites(const std::vector<int>& map_site2msphere,
                                     const std::set<int>& spheres_and_1rings);
  void update_convex_cells_voro_and_tet_ids(
      const std::vector<int>& map_site2msphere,
      const std::vector<int>& map_tet_new2orig,
      std::vector<ConvexCellHost>& cells_partials, bool is_debug);

  std::vector<ConvexCellHost> merge_convex_cells(
      const std::set<int>& valid_sphere_ids,
      const std::set<int>& spheres_and_1rings,
      const std::vector<ConvexCellHost>& convex_cells_prev,
      const std::vector<ConvexCellHost>& convex_cells_new, bool is_debug);

  void load_partial_tet_given_spheres(
      const std::vector<int>& old_tet_indices,
      const std::vector<int>& old_tet_fids,
      const std::vector<int>& old_tet_f_adjs,
      const std::set<int>& given_spheres,
      const std::vector<MedialSphere>& all_medial_spheres,
      std::vector<int>& partial_tet_indices, std::vector<int>& map_tet_new2orig,
      std::vector<int>& partial_tet_fids, std::vector<int>& partial_tet_f_adjs,
      bool is_debug);

 public:
  std::vector<ConvexCellHost> powercells;
  bool is_debug = false;

 public:
  const TetMesh* tet_mesh;
  const SurfaceMesh* sf_mesh;
  const Parameter* params;
  std::vector<MedialSphere>* all_medial_spheres;

  bool site_is_transposed;
  bool is_given_all_tets;

  RegularTriangulationNN rt;
  std::vector<float> site;
  std::vector<float> site_weights;
  std::vector<uint> site_flags;
  std::vector<int> site_knn;
  std::vector<float> site_cell_vol;

  int num_itr_rpd;
  int n_site;  // number of sites
  int site_k;  // maximum number of neighboring spheres

 public:
  // ---------------------------------------------------------------------
  // MSD_SPH_ANISO -- route-1 anisotropic (spherical-harmonic) power diagram.
  // Published once per diagram rebuild by publishShToRpd()
  // (MATStruct src/unified_ps/ups_sph_loop.cpp) and consumed by
  // rpd3d_compute.cxx + the CUDA kernel. sph_stride == 0 means anisotropy is
  // OFF and every path below is bit-for-bit the isotropic one.
  //
  // Sized by the SITE count, never by the AtomSet: the chassis can add spheres
  // between rebuilds and the kernel indexes these by site id, so a short array
  // is an out-of-bounds read rather than a fallback. Sites past the AtomSet get
  // sph_l = 0 and sph_rmax = their own radius, i.e. exactly isotropic.

  // n_site * sph_stride real-SH coefficients, fp32, in realSH's own
  // out[l*(l+1)+m] order. Row i is site i's radius function r_i(d).
  std::vector<float> sph_coeffs;
  // Per-site SH band actually published (0 = isotropic site). <= the L cap.
  std::vector<int> sph_l;
  // sph_stride normalization constants sqrt((2l+1)/4pi * (l-|m|)!/(l+|m|)!),
  // same order as sph_coeffs; the device evaluator indexes it directly.
  std::vector<float> sph_nrm;
  // Per-site analytic sup of r_i(d), from |Y_lm| <= sqrt((2l+1)/4pi):
  // r_max = sum_k |c_k| sqrt((2 l_k + 1)/4pi). MUST be an upper bound -- the
  // CGAL RT and the GPU candidate sets are built on r_max^2 and stop being
  // supersets if it is under-estimated (cells then truncate silently).
  std::vector<float> sph_rmax;
  // shDim(L cap) = coefficients per site. 0 disables every SH path.
  int sph_stride = 0;
  // Global max over sites of (r_max^2 - r_min^2). The tet<->sphere relation is
  // widened by this, or an anisotropic pair is rejected before any cell exists.
  float sph_gap = 0.f;

  // MSD_SPH_ANISO_RING: snapshot of the RT 1-ring (site_knn / site_k) taken
  // before expandSiteKnnRing() grows the candidate set, kept for the face
  // census so it can tell a genuine 1-ring neighbour from a ring-expansion one.
  std::vector<int> sph_knn_ring1;
  int sph_k_ring1 = 0;
};
