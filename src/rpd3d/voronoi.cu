#include <math.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <unordered_set>

#include "convex_cell.h"
#include "kNN-CUDA/knncuda.h"
#include "stopwatch.h"
#include "voronoi.h"

// ###################  Status   ######################
char StatusStr[7][128] = {"triangle_overflow",
                          "vertex_overflow",
                          "inconsistent_boundary",
                          "security_radius_not_reached",
                          "success",
                          "needs_exact_predicates",
                          "no_intersection"};

void show_status_stats(std::vector<Status>& stat) {
  IF_VERBOSE(
      std::cerr << " \n\n\n---------Summary of success/failure------------\n");
  std::vector<int> nb_statuss(7, 0);
  FOR(i, stat.size()) nb_statuss[stat[i]]++;
  IF_VERBOSE(FOR(r, 7) std::cerr << " " << StatusStr[r] << "   "
                                 << nb_statuss[r] << "\n";)
  std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  "
            << stat.size() << "\n";
}

void cuda_check_error() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// ###################  Functions   ######################
__global__ void compute_new_site(float* site, int n_site, size_t site_pitch,
                                 float* cell_bary_sum,
                                 const size_t cell_bary_sum_pitch,
                                 float* cell_vol) {
  int seed = blockIdx.x * blockDim.x + threadIdx.x;
  if (seed >= n_site) return;

  if (cell_vol[seed] != 0) {
    if (site_pitch) {
      site[seed] = cell_bary_sum[seed] / cell_vol[seed];
      site[seed + site_pitch] =
          cell_bary_sum[seed + cell_bary_sum_pitch] / cell_vol[seed];
      site[seed + (site_pitch << 1)] =
          cell_bary_sum[seed + (cell_bary_sum_pitch << 1)] / cell_vol[seed];
    } else {
      site[3 * seed] = cell_bary_sum[3 * seed] / cell_vol[seed];
      site[3 * seed + 1] = cell_bary_sum[3 * seed + 1] / cell_vol[seed];
      site[3 * seed + 2] = cell_bary_sum[3 * seed + 2] / cell_vol[seed];
    }

    // if (seed == 588) {
    // printf(
    //     "seed %d has cell_bary_sum: (%f,%f,%f), cell_vol: %f, site: "
    //     "(%f,%f,%f) \n",
    //     seed, cell_bary_sum[seed], cell_bary_sum[seed + cell_bary_sum_pitch],
    //     cell_bary_sum[seed + (cell_bary_sum_pitch << 1)], cell_vol[seed],
    //     site[seed], site[seed + site_pitch], site[seed + (site_pitch << 1)]);
    // }
  }
}

__global__ void compute_tet_centroid(const float* vert, const int n_vert,
                                     const size_t vert_pitch, const int* idx,
                                     const int n_tet, const size_t idx_pitch,
                                     float* tet_centroid,
                                     const size_t tet_centroid_pitch) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_tet) return;

  float3 centroid = {0.0f, 0.0f, 0.0f};

  FOR(i, 4) {
    centroid.x += vert[idx[tid + i * idx_pitch]];
    centroid.y += vert[idx[tid + i * idx_pitch] + vert_pitch];
    centroid.z += vert[idx[tid + i * idx_pitch] + (vert_pitch << 1)];
  }

  centroid.x *= 0.25f;
  centroid.y *= 0.25f;
  centroid.z *= 0.25f;

  tet_centroid[tid] = centroid.x;
  tet_centroid[tid + tet_centroid_pitch] = centroid.y;
  tet_centroid[tid + (tet_centroid_pitch << 1)] = centroid.z;
}

__global__ void transpose_site(const float* site, const int n_site,
                               float* site_transposed,
                               const size_t site_transposed_pitch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  site_transposed[idx] = site[3 * idx];
  site_transposed[idx + site_transposed_pitch] = site[3 * idx + 1];
  site_transposed[idx + (site_transposed_pitch << 1)] = site[3 * idx + 2];
}

// ninwang:
// for each tet centroid, find n_tet number of nearest weighte sites
void compute_tet_weighted_knn_dev(const float* vert_dev, const int n_vert,
                                  const size_t vert_pitch, const int* idx_dev,
                                  const int n_tet, const size_t idx_pitch,
                                  const float* site_dev, const int n_site,
                                  const size_t site_pitch,
                                  const float* site_weights_dev,
                                  int* tet_knn_dev, const size_t tet_knn_pitch,
                                  const int tet_k) {
  float* tet_centroid_dev = nullptr;
  size_t tet_centroid_pitch_in_bytes;
  cudaMallocPitch((void**)&tet_centroid_dev, &tet_centroid_pitch_in_bytes,
                  n_tet * sizeof(float), 3);
  cuda_check_error();
  size_t tet_centroid_pitch = tet_centroid_pitch_in_bytes / sizeof(float);

  compute_tet_centroid<<<n_tet / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
      vert_dev, n_vert, vert_pitch, idx_dev, n_tet, idx_pitch, tet_centroid_dev,
      tet_centroid_pitch);

  if (site_pitch)  // has been transposed
    knn_weighted_cuda_global_dev(site_dev, n_site, site_pitch, site_weights_dev,
                                 tet_centroid_dev, n_tet, tet_centroid_pitch, 3,
                                 tet_k, tet_knn_dev, tet_knn_pitch);
  else {
    float* site_transposed_dev = nullptr;
    size_t site_transposed_pitch_in_bytes;
    cudaMallocPitch((void**)&site_transposed_dev,
                    &site_transposed_pitch_in_bytes, n_site * sizeof(float), 3);
    cuda_check_error();
    size_t site_transposed_pitch =
        site_transposed_pitch_in_bytes / sizeof(float);
    transpose_site<<<n_site / KNN_BLOCK_SIZE + 1, KNN_BLOCK_SIZE>>>(
        site_dev, n_site, site_transposed_dev, site_transposed_pitch);

    knn_weighted_cuda_global_dev(site_transposed_dev, n_site,
                                 site_transposed_pitch, site_weights_dev,
                                 tet_centroid_dev, n_tet, tet_centroid_pitch, 3,
                                 tet_k, tet_knn_dev, tet_knn_pitch);

    cudaFree(site_transposed_dev);
  }

  cudaFree(tet_centroid_dev);
}

// return tet_sphere_relate_dev
// size #spheres x #tets
__global__ void tet_sphere_relations_dev(
    const int n_vert, const int* idx_dev, const int n_tet,
    const size_t idx_pitch, const int n_site, const uint* site_flags_dev,
    const int* site_knn_dev, const size_t site_knn_pitch, const int site_k,
    const float* pdist_dev, const size_t pdist_pitch,
    int* tet_sphere_relate_dev, size_t tet_sphere_relate_pitch,
    const float sph_gap) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_tet) return;

  // The tet's 4 vertex ids do not depend on the site -- load them once instead
  // of re-reading idx_dev inside the (n_site x site_k) double loop.
  int v_ids[4];
  FOR(t, 4) v_ids[t] = idx_dev[tid + t * idx_pitch];

  // each thread handles all n_site
  FOR(si_idx, n_site) {
    // case 1: if si is not selected, then tet is not related to si
    if (site_flags_dev[si_idx] != SiteFlag::is_selected) {
      tet_sphere_relate_dev[tid + si_idx * tet_sphere_relate_pitch] = 0;
      continue;
    }
    // case 2: find all neighboring site of sphere i (see note)
    //
    // pd(v, si) is invariant across the neighbour loop -- hoist the 4 loads
    // out of it (they were re-read site_k times each).
    float pd_i[4];
    FOR(t, 4) pd_i[t] = pdist_dev[v_ids[t] + si_idx * pdist_pitch];

    // The original accumulated num_relate_hp and site_real_k over ALL site_k
    // neighbours, then set is_relate = (num_relate_hp == site_real_k). Both
    // counters rise by at most 1 per neighbour and num_relate_hp only rises
    // when the neighbour passes, so the FIRST failing neighbour opens a gap
    // that can never close: the result is already decided as 0. Bailing there
    // is exactly equivalent, and the relation is ~0.06% dense, so almost every
    // (tet, site) pair now exits after a neighbour or two instead of 62.
    int is_relate = 1;
    FOR(sm, site_k) {
      int sm_idx = site_knn_dev[si_idx + sm * site_knn_pitch];
      if (sm_idx == -1) continue;  // some may be -1
      bool any_closer = false;
      FOR(t, 4) {
        // tet vertex v_ids[t] closer to sphere i than j
        // MSD_SPH_ANISO: the pdist matrix is built on the SCALAR weight
        // r_max^2, but the anisotropic bisector can sit up to
        // (r_max_j^2 - r_min_j^2) / (2 D_ij) away from where that puts it. Slack
        // the NEIGHBOUR's distance by the global gap so a tet that an
        // anisotropic sphere can still reach is not dropped from the relation
        // before any cell exists -- which is why widening site_knn alone could
        // not help. sph_gap == 0 restores the exact original comparison.
        if (pdist_dev[v_ids[t] + sm_idx * pdist_pitch] + sph_gap > pd_i[t]) {
          any_closer = true;
          break;
        }
      }  // 4 tet vertices
      if (!any_closer) {
        is_relate = 0;
        break;
      }
    }  // for site_k (all neighbor spheres)

    tet_sphere_relate_dev[tid + si_idx * tet_sphere_relate_pitch] = is_relate;
  }  // for n_site
}

// Device-side helpers for compute_tet_sphere_relation: the tet-to-sphere
// relation is built directly on the GPU (ascending sid per tet = same order as
// the old host loop, so the result is bit-identical). This removes a
// #site x #tet D2H copy and a single-threaded O(#site * #tet) host loop per
// RPD call.
__global__ void count_tet_related_spheres(const int* rel,
                                          const size_t rel_pitch,
                                          const int n_tet, const int n_site,
                                          int* counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_tet) return;
  int c = 0;
  for (int sid = 0; sid < n_site; ++sid)
    if (rel[sid * rel_pitch + tid] == 1) ++c;
  counts[tid] = c;
}

// CSR fill. Slot s = tet_offsets[tid] + k holds the k-th related sphere of tet
// tid, in ascending sid -- exactly the content the dense layout put at
// tet_knn[k][tid], minus the -1 padding. slot2tet[s] = tid lets the RPD kernel
// recover its tet from a flat slot index without a search.
//
// The dense layout reserved max-over-ALL-tets slots for EVERY tet, so one
// locally dense cluster of spheres charged the whole mesh: 146k tets x max 16
// x 3456 B/ConvexCellTransfer = 8.1 GB of RPD output buffer on a 6 GB card.
__global__ void fill_tet_knn_csr_dev(const int* rel, const size_t rel_pitch,
                                     const int n_tet, const int n_site,
                                     const int* tet_offsets, int* tet_knn_csr,
                                     int* slot2tet) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_tet) return;
  int s = tet_offsets[tid];
  const int end = tet_offsets[tid + 1];
  for (int sid = 0; sid < n_site && s < end; ++sid)
    if (rel[sid * rel_pitch + tid] == 1) {
      tet_knn_csr[s] = sid;
      slot2tet[s] = tid;
      ++s;
    }
}

// Stable stream compaction of the per-(tet,seed) convex-cell output: the old
// path value-initialized a #slots-sized host vector (~GBs: sizeof
// ConvexCellTransfer is ~2.7KB) and copied EVERY slot D2H, only for the host
// to skip the invalid majority. Flag validity on device (same predicate as
// is_convex_cell_valid), exclusive-scan the flags (tiny host scan), and gather
// valid cells in slot order -- the host then sees the exact same valid cells
// in the exact same order as before, so downstream dedup is bit-identical.
__global__ void flag_valid_cells(const ConvexCellTransfer* cells, const int n,
                                 int* flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Status s = cells[i].status;
  flags[i] =
      (s == Status::success || s == Status::security_radius_not_reached) ? 1
                                                                         : 0;
}

__global__ void gather_valid_cells(const ConvexCellTransfer* cells,
                                   const int* flags, const int* pos,
                                   const int n, ConvexCellTransfer* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (flags[i]) out[pos[i]] = cells[i];
}

// ninwang
// update tet_knn and tet_k
// use power distance to find the potential relations between tet and sphere
void compute_tet_sphere_relation(
    const float* vert_dev, const int n_vert, const size_t vert_pitch,
    const int* idx_dev, const int n_tet, const size_t idx_pitch,
    const float* site_dev, const uint* site_flags_dev, const int n_site,
    const size_t site_pitch, const float* site_weights_dev,
    const int* site_knn_dev, const int site_k, const size_t site_knn_pitch,
    float* tet_pdist_dev, const size_t tet_pdist_pitch,
    int* tet_sphere_relate_dev, const size_t tet_sphere_relate_pitch,
    int& tet_k, std::vector<int>& tet_counts, const float sph_gap) {
  // printf("calling compute_tet_sphere_relation... \n");

  // Opt-in sub-attribution of this function (it is ~87% of the RPD phase on
  // large CAD parts). Each mark syncs the device first, so the numbers are
  // real per-step costs rather than async launch times -- which is why it is
  // OFF unless MSD_PROF_TETSPHERE is set.
  static const bool s_profTS = std::getenv("MSD_PROF_TETSPHERE") != nullptr;
  auto ts_now = [&]() {
    if (s_profTS) cudaDeviceSynchronize();
    return std::chrono::steady_clock::now();
  };
  auto ts_t0 = ts_now();
  auto ts_mark = [&](const char* what) {
    if (!s_profTS) return;
    const auto t = ts_now();
    printf("[tet_sphere] %-18s %7.3f s\n", what,
           std::chrono::duration<double>(t - ts_t0).count());
    ts_t0 = t;
  };

  assert(site_pitch > 0);  // always transposed
  // step 1: compute power distancese for each tet vertex
  // (the #spheres x #tet_vertices pdist matrix and the #spheres x #tets
  // relation matrix are now owned by the caller's grow-only device cache --
  // they used to be cudaMallocPitch'd and cudaFree'd on EVERY call, which on a
  // large CAD part is ~2.5 GB of allocator churn per RPD call, and cudaFree
  // synchronises the device)
  ts_mark("entry");
  power_dist_cuda_global_dev(site_dev, n_site, site_pitch, site_weights_dev,
                             vert_dev, n_vert, vert_pitch, 3, tet_pdist_dev,
                             tet_pdist_pitch);
  ts_mark("kernel_pdist");

  // (debug D2H copy of the full tet_pdist matrix removed -- it fed only
  // commented-out prints and cost a ~15MB transfer per RPD call)

  // printf("tet_pdist matrix: \n\t");
  // for (uint vid = 0; vid < n_vert; vid++) {    // column
  //   for (uint sid = 0; sid < n_site; sid++) {  // row
  //     printf("%f, ", tet_pdist[vid + sid * n_vert]);
  //   }
  //   printf("\n\t ");
  // }
  // printf("\n");
  // printf("done power_dist_cuda_global_dev \n");

  // step 2: compute tet-sphere relation matrix
  // size: #spheres x #tets
  // value:
  // 1: tet relates to sphere
  // 0: not relate
  tet_sphere_relations_dev<<<n_tet / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
      n_vert, idx_dev, n_tet, idx_pitch, n_site, site_flags_dev, site_knn_dev,
      site_knn_pitch, site_k, tet_pdist_dev, tet_pdist_pitch,
      tet_sphere_relate_dev, tet_sphere_relate_pitch, sph_gap);
  ts_mark("kernel_relate");

  // Per-tet count of related spheres, computed on device: counts kernel + tiny
  // (#tet ints) D2H. The relation matrix itself stays on the GPU for
  // fill_tet_knn_csr_dev. tet_counts drives the CSR slot layout; tet_k (the
  // max) is now reported only -- it no longer sizes anything, because charging
  // EVERY tet the global maximum is what made the RPD buffer O(n_tet * max).
  int* counts_dev = nullptr;
  cudaMalloc((void**)&counts_dev, n_tet * sizeof(int));
  cuda_check_error();
  count_tet_related_spheres<<<(n_tet + 255) / 256, 256>>>(
      tet_sphere_relate_dev, tet_sphere_relate_pitch, n_tet, n_site,
      counts_dev);
  cuda_check_error();
  tet_counts.resize(n_tet);
  cudaMemcpy(tet_counts.data(), counts_dev, n_tet * sizeof(int),
             cudaMemcpyDeviceToHost);
  cuda_check_error();
  cudaFree(counts_dev);
  tet_k = 0;
  for (int tid = 0; tid < n_tet; tid++)
    if (tet_counts[tid] > tet_k) tet_k = tet_counts[tid];
  printf("updated tet_k: %d\n", tet_k);
  ts_mark("count+D2H+max");
  // No clean-up: both matrices belong to the caller's device cache, which
  // reuses them across calls and releases them in cleanup_voronoi_gpu_cache()
  // at the end of the particle stage.
}

void copy_tet_data(const std::vector<float>& vertices,
                   const std::vector<int>& indices, float*& vert_dev,
                   size_t& vert_pitch, int*& idx_dev, size_t& idx_pitch) {
  size_t n_vert = vertices.size() / 3, n_tet = (indices.size() >> 2);

  // transpose
  float* vert_T = new float[3 * n_vert];
  int* idx_T = new int[n_tet << 2];
  FOR(i, n_vert) {
    vert_T[i] = vertices[3 * i];
    vert_T[i + n_vert] = vertices[3 * i + 1];
    vert_T[i + (n_vert << 1)] = vertices[3 * i + 2];
  }
  FOR(i, n_tet) {
    idx_T[i] = indices[(i << 2)];
    idx_T[i + n_tet] = indices[(i << 2) + 1];
    idx_T[i + (n_tet << 1)] = indices[(i << 2) + 2];
    idx_T[i + n_tet * 3] = indices[(i << 2) + 3];
  }

  size_t vert_pitch_in_bytes, idx_pitch_in_bytes;
  cudaMallocPitch((void**)&vert_dev, &vert_pitch_in_bytes,
                  n_vert * sizeof(float), 3);
  cuda_check_error();
  cudaMallocPitch((void**)&idx_dev, &idx_pitch_in_bytes, n_tet * sizeof(int),
                  4);
  cuda_check_error();
  vert_pitch = vert_pitch_in_bytes / sizeof(float);
  idx_pitch = idx_pitch_in_bytes / sizeof(int);
  cudaMemcpy2D(vert_dev, vert_pitch_in_bytes, vert_T, n_vert * sizeof(float),
               n_vert * sizeof(float), 3, cudaMemcpyHostToDevice);
  cuda_check_error();
  cudaMemcpy2D(idx_dev, idx_pitch_in_bytes, idx_T, n_tet * sizeof(int),
               n_tet * sizeof(int), 4, cudaMemcpyHostToDevice);
  cuda_check_error();

  delete[] vert_T;
  delete[] idx_T;
}

/**
 * See load_tet_adj_info()
 *
 * 1. We store the number of shared adjacent cells for furture calculation of
 * Euler.
 * 2. We also stores unique id (int2) for each face, defined by (fid, -1)
 *
 * Vertex Eulers are stored as indices
 * Euler(v) = 1. / (#adjacent cells)
 *
 * Edge Eulers are stored as a diagonal adjacency matrix
 *
 * Face Eulers (originally from tets) are always 1/2 (adjacent to 2 cells)
 * except boundary
 */
void load_num_adjacent_cells_and_ids(
    const std::vector<int>& v_adjs, const std::vector<int>& e_adj_offsets,
    const std::vector<int>& e_adj_neighbors,
    const std::vector<int>& e_adj_vals, const std::vector<int>& f_adjs,
    const std::vector<int>& f_ids, int*& v_adjs_dev, int*& e_adj_offsets_dev,
    int*& e_adj_neighbors_dev, int*& e_adj_vals_dev, int*& f_adjs_dev,
    int*& f_ids_dev) {
  assert(!v_adjs.empty() && !e_adj_neighbors.empty() && !f_adjs.empty() &&
         !f_ids.empty());
  std::cout << "loaded #adjacent cells for v: " << v_adjs.size()
            << ", and e: " << e_adj_neighbors.size() << ", and f "
            << f_adjs.size() << std::endl;

  // ninwang: cuda need a pointer to pointer
  cudaMalloc((void**)&v_adjs_dev, v_adjs.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(v_adjs_dev, v_adjs.data(), v_adjs.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cuda_check_error();

  cudaMalloc((void**)&e_adj_offsets_dev, e_adj_offsets.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(e_adj_offsets_dev, e_adj_offsets.data(),
             e_adj_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
  cuda_check_error();

  cudaMalloc((void**)&e_adj_neighbors_dev,
             e_adj_neighbors.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(e_adj_neighbors_dev, e_adj_neighbors.data(),
             e_adj_neighbors.size() * sizeof(int), cudaMemcpyHostToDevice);
  cuda_check_error();

  cudaMalloc((void**)&e_adj_vals_dev, e_adj_vals.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(e_adj_vals_dev, e_adj_vals.data(),
             e_adj_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
  cuda_check_error();

  cudaMalloc((void**)&f_adjs_dev, f_adjs.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(f_adjs_dev, f_adjs.data(), f_adjs.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cuda_check_error();

  cudaMalloc((void**)&f_ids_dev, f_ids.size() * sizeof(int));
  cuda_check_error();
  cudaMemcpy(f_ids_dev, f_ids.data(), f_ids.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cuda_check_error();
}

__host__ __device__ cuchar3 convert(uchar3 orig) {
  return cmake_uchar3(orig.x, orig.y, orig.z);
}
__host__ __device__ cuchar4 convert(uchar4 orig) {
  return cmake_uchar4(orig.x, orig.y, orig.z, orig.w);
}
__host__ __device__ cint2 convert(int2 orig) {
  return cmake_int2(orig.x, orig.y);
}
__host__ __device__ cfloat4 convert(float4 orig) {
  return cmake_float4(orig.x, orig.y, orig.z, orig.w);
}
__host__ __device__ cfloat5 convert(float5 orig) {
  return cmake_float5(orig.x, orig.y, orig.z, orig.w, orig.h);
}

void copy_cc(const ConvexCellTransfer& cc, ConvexCellHost& cc_trans) {
  cc_trans.is_active = true;
  cc_trans.status = cc.status;
  cc_trans.thread_id = cc.thread_id;
  cc_trans.voro_id = cc.voro_id;
  cc_trans.tet_id = cc.tet_id;
  cc_trans.euler = cc.euler;
  cc_trans.weight = cc.weight;
  cc_trans.nb_v = cc.nb_v;
  cc_trans.nb_p = cc.nb_p;
  cc_trans.nb_e = cc.nb_e;
  FOR(i, cc.nb_v) cc_trans.ver_data_trans[i] = convert(cc.ver_data_trans[i]);
  FOR(i, cc.nb_p) cc_trans.clip_data_trans[i] = convert(cc.clip_data_trans[i]);
  FOR(i, cc.nb_p)
  cc_trans.clip_id2_data_trans[i] = convert(cc.clip_id2_data_trans[i]);
  FOR(i, cc.nb_e) cc_trans.edge_data[i] = convert(cc.edge_data[i]);
}

/**
 * vertices: tet vertices
 * indices: tet 4 indices of vertices [can be parital indices]
 */
namespace {
// ---------------------------------------------------------------------------
// Persistent device-buffer cache for compute_clipped_voro_diagram_GPU().
//
// The function runs O(100x) per pipeline run with a FIXED tet mesh; only the
// sphere/site data changes. Fresh cudaMalloc/cudaFree of ~30 buffers every call
// dominated wall time (~1.3 s/call overhead vs ~60 ms of actual GPU compute --
// measured). We keep the buffers alive and grow-only across calls instead.
//
// Reuse is semantically identical to fresh allocation: cudaMalloc never zeroes
// memory, so a sufficiently large reused buffer behaves like a new one, and
// every explicit cudaMemset in the function is preserved. For pitched buffers
// we reuse the original pitch (valid while logical width/height stay within the
// allocated capacity).
//
// Assumes serial invocation (compute_rpd() is called serially); the cache is
// process-global and not thread-safe.
// ---------------------------------------------------------------------------
struct VoroDevCache {
  struct Buf {
    void* p = nullptr;
    size_t cap = 0;  // bytes
  };
  struct PitchBuf {
    void* p = nullptr;
    size_t cap_w = 0;  // allocated row width in bytes
    size_t cap_h = 0;  // allocated rows
    size_t pitch = 0;  // bytes
  };

  void* ensure(Buf& b, size_t need_bytes) {
    if (need_bytes > b.cap) {
      if (b.p) cudaFree(b.p);
      cudaMalloc(&b.p, need_bytes);
      cuda_check_error();
      b.cap = need_bytes;
    }
    return b.p;
  }
  // returns pitch in bytes; device pointer is b.p
  size_t ensure_pitch(PitchBuf& b, size_t width_bytes, size_t rows) {
    if (!b.p || width_bytes > b.cap_w || rows > b.cap_h) {
      if (b.p) cudaFree(b.p);
      size_t w = width_bytes > b.cap_w ? width_bytes : b.cap_w;
      size_t h = rows > b.cap_h ? rows : b.cap_h;
      cudaMallocPitch(&b.p, &b.pitch, w, h);
      cuda_check_error();
      b.cap_w = w;
      b.cap_h = h;
    }
    return b.pitch;
  }

  // static tet mesh (uploaded once; rebuilt only if mesh dimensions change)
  int cached_n_vert = -1;
  int cached_n_tet = -1;
  float* vert_dev = nullptr;
  int* idx_dev = nullptr;
  size_t vert_pitch = 0, idx_pitch = 0;  // in elements
  int* v_adjs_dev = nullptr;
  int* e_adj_offsets_dev = nullptr;
  int* e_adj_neighbors_dev = nullptr;
  int* e_adj_vals_dev = nullptr;
  int* f_adjs_dev = nullptr;
  int* f_ids_dev = nullptr;

  // per-call work buffers (grow-only)
  Buf voronoi_cells, cell_vol, site_weights, site_flags, convex_cells,
      cell_bary_sum_lin, cc_flags, cc_pos, cc_compact, tet_offsets,
      tet_knn_csr, slot2tet;
  // MSD_SPH_ANISO: uploaded beside site_weights, freed beside it. Grow-only
  // like the rest, so an anisotropy-off run never allocates them.
  Buf sph_coeffs, sph_l, sph_nrm;
  PitchBuf cell_bary_sum, site_transposed, site_knn, tet_pdist, tet_relate;

  // Free every device buffer and reset to pristine state. A subsequent
  // compute_clipped_voro_diagram_GPU() re-allocates on demand and re-uploads
  // the tet mesh (cached_n_vert/tet reset forces the rebuild), so this is
  // context-preserving.
  void free_all() {
    auto freeBuf = [](Buf& b) {
      if (b.p) cudaFree(b.p);
      b.p = nullptr;
      b.cap = 0;
    };
    auto freePitch = [](PitchBuf& b) {
      if (b.p) cudaFree(b.p);
      b.p = nullptr;
      b.cap_w = 0;
      b.cap_h = 0;
      b.pitch = 0;
    };
    if (vert_dev) cudaFree(vert_dev);
    if (idx_dev) cudaFree(idx_dev);
    if (v_adjs_dev) cudaFree(v_adjs_dev);
    if (e_adj_offsets_dev) cudaFree(e_adj_offsets_dev);
    if (e_adj_neighbors_dev) cudaFree(e_adj_neighbors_dev);
    if (e_adj_vals_dev) cudaFree(e_adj_vals_dev);
    if (f_adjs_dev) cudaFree(f_adjs_dev);
    if (f_ids_dev) cudaFree(f_ids_dev);
    vert_dev = nullptr;
    idx_dev = nullptr;
    v_adjs_dev = e_adj_offsets_dev = e_adj_neighbors_dev = e_adj_vals_dev =
        f_adjs_dev = f_ids_dev = nullptr;
    vert_pitch = idx_pitch = 0;
    cached_n_vert = cached_n_tet = -1;
    freeBuf(voronoi_cells);
    freeBuf(cell_vol);
    freeBuf(site_weights);
    freeBuf(site_flags);
    freeBuf(sph_coeffs);
    freeBuf(sph_l);
    freeBuf(sph_nrm);
    freeBuf(convex_cells);
    freeBuf(cell_bary_sum_lin);
    freeBuf(cc_flags);
    freeBuf(cc_pos);
    freeBuf(cc_compact);
    freeBuf(tet_offsets);
    freeBuf(tet_knn_csr);
    freeBuf(slot2tet);
    freePitch(cell_bary_sum);
    freePitch(site_transposed);
    freePitch(site_knn);
    freePitch(tet_pdist);
    freePitch(tet_relate);
  }
};
VoroDevCache g_voro;
}  // namespace

void cleanup_voronoi_gpu_cache() { g_voro.free_all(); }

std::vector<ConvexCellHost> compute_clipped_voro_diagram_GPU(
    const int num_itr_global, const std::vector<float>& vertices,
    const std::vector<int>& indices, const std::map<int, std::set<int>>& v2tets,
    const std::vector<int>& v_adjs, const std::vector<int>& e_adj_offsets,
    const std::vector<int>& e_adj_neighbors, const std::vector<int>& e_adj_vals,
    const std::vector<int>& f_adjs, const std::vector<int>& f_ids,
    std::vector<float>& site, const int n_site,
    const std::vector<float>& site_weights, const std::vector<uint>& site_flags,
    const std::vector<int>& site_knn, const int site_k,
    std::vector<float>& site_cell_vol, const bool site_is_transposed,
    int nb_Lloyd_iter, int preferred_tet_k,
    const std::vector<float>* sph_coeffs, const std::vector<int>* sph_l,
    const std::vector<float>* sph_nrm, int sph_stride, float sph_gap) {
  cudaSetDevice(0);  // specify a device to be used for GPU computation
  int n_vert = vertices.size() / 3;
  int n_tet = (indices.size() >> 2);

  // The tet mesh is static across calls -- upload it once and reuse. Rebuild
  // only if the mesh dimensions change (e.g. a different model in-process).
  if (g_voro.cached_n_vert != n_vert || g_voro.cached_n_tet != n_tet) {
    if (g_voro.vert_dev) cudaFree(g_voro.vert_dev);
    if (g_voro.idx_dev) cudaFree(g_voro.idx_dev);
    if (g_voro.v_adjs_dev) cudaFree(g_voro.v_adjs_dev);
    if (g_voro.e_adj_offsets_dev) cudaFree(g_voro.e_adj_offsets_dev);
    if (g_voro.e_adj_neighbors_dev) cudaFree(g_voro.e_adj_neighbors_dev);
    if (g_voro.e_adj_vals_dev) cudaFree(g_voro.e_adj_vals_dev);
    if (g_voro.f_adjs_dev) cudaFree(g_voro.f_adjs_dev);
    if (g_voro.f_ids_dev) cudaFree(g_voro.f_ids_dev);
    copy_tet_data(vertices, indices, g_voro.vert_dev, g_voro.vert_pitch,
                  g_voro.idx_dev, g_voro.idx_pitch);
    load_num_adjacent_cells_and_ids(
        v_adjs, e_adj_offsets, e_adj_neighbors, e_adj_vals, f_adjs, f_ids,
        g_voro.v_adjs_dev, g_voro.e_adj_offsets_dev, g_voro.e_adj_neighbors_dev,
        g_voro.e_adj_vals_dev, g_voro.f_adjs_dev, g_voro.f_ids_dev);
    g_voro.cached_n_vert = n_vert;
    g_voro.cached_n_tet = n_tet;
  }
  float* vert_dev = g_voro.vert_dev;
  int* idx_dev = g_voro.idx_dev;
  size_t vert_pitch = g_voro.vert_pitch, idx_pitch = g_voro.idx_pitch;
  int* v_adjs_dev = g_voro.v_adjs_dev;
  int* e_adj_offsets_dev = g_voro.e_adj_offsets_dev;
  int* e_adj_neighbors_dev = g_voro.e_adj_neighbors_dev;
  int* e_adj_vals_dev = g_voro.e_adj_vals_dev;
  int* f_adjs_dev = g_voro.f_adjs_dev;
  int* f_ids_dev = g_voro.f_ids_dev;
  assert(v_adjs.size() == n_vert);

  // allocate memory for voronoi cell
  VoronoiCell* voronoi_cells_dev = (VoronoiCell*)g_voro.ensure(
      g_voro.voronoi_cells, n_site * sizeof(VoronoiCell));

  // allocate memory forninwang:   output points
  float* cell_bary_sum_dev = nullptr;
  size_t cell_bary_sum_pitch_in_bytes = 0, cell_bary_sum_pitch = 0;
  if (site_is_transposed) {
    cell_bary_sum_pitch_in_bytes =
        g_voro.ensure_pitch(g_voro.cell_bary_sum, n_site * sizeof(float), 3);
    cell_bary_sum_dev = (float*)g_voro.cell_bary_sum.p;
    cell_bary_sum_pitch = cell_bary_sum_pitch_in_bytes / sizeof(float);
  } else {
    cell_bary_sum_dev = (float*)g_voro.ensure(g_voro.cell_bary_sum_lin,
                                              3 * n_site * sizeof(float));
  }

  // allocate memory for cell volume
  site_cell_vol.clear();
  site_cell_vol.resize(n_site);
  float* cell_vol_dev =
      (float*)g_voro.ensure(g_voro.cell_vol, n_site * sizeof(float));

  // ninwang: allocate memory for site weights
  assert(site_weights.size() == n_site);
  float* site_weights_dev =
      (float*)g_voro.ensure(g_voro.site_weights, n_site * sizeof(float));
  cudaMemcpy(site_weights_dev, site_weights.data(), n_site * sizeof(float),
             cudaMemcpyHostToDevice);

  // MSD_SPH_ANISO: the per-site radius functions, uploaded beside the scalar
  // weights and bound at the launch below. Anything missing or short leaves
  // sh_stride_dev == 0, and the kernel is then the isotropic one byte for byte.
  const float* sph_coeffs_dev = nullptr;
  const int* sph_l_dev = nullptr;
  const float* sph_nrm_dev = nullptr;
  int sh_stride_dev = 0;
  if (sph_stride > 0 && sph_coeffs && sph_l && sph_nrm &&
      (int)sph_l->size() >= n_site &&
      (long long)sph_coeffs->size() >= (long long)n_site * sph_stride &&
      (int)sph_nrm->size() >= sph_stride) {
    sh_stride_dev = sph_stride;
    float* c = (float*)g_voro.ensure(g_voro.sph_coeffs,
                                     (size_t)n_site * sph_stride * sizeof(float));
    int* l = (int*)g_voro.ensure(g_voro.sph_l, (size_t)n_site * sizeof(int));
    float* nr =
        (float*)g_voro.ensure(g_voro.sph_nrm, (size_t)sph_stride * sizeof(float));
    cudaMemcpy(c, sph_coeffs->data(),
               (size_t)n_site * sph_stride * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(l, sph_l->data(), (size_t)n_site * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(nr, sph_nrm->data(), (size_t)sph_stride * sizeof(float),
               cudaMemcpyHostToDevice);
    cuda_check_error();
    sph_coeffs_dev = c;
    sph_l_dev = l;
    sph_nrm_dev = nr;
  }
  cuda_check_error();

  // ninwang: allocate memory for site flag
  assert(site_flags.size() == n_site);
  uint* site_flags_dev =
      (uint*)g_voro.ensure(g_voro.site_flags, n_site * sizeof(uint));
  cudaMemcpy(site_flags_dev, site_flags.data(), n_site * sizeof(uint),
             cudaMemcpyHostToDevice);
  cuda_check_error();

  // allocate memory for site and site knn
  assert(site_knn.size() == (site_k + 1) * n_site);
  float* site_transposed_dev = nullptr;
  int* site_knn_dev = nullptr;
  size_t site_pitch = 0, site_pitch_in_bytes = 0, site_knn_pitch = 0,
         site_knn_pitch_in_bytes = 0;

  //////////////////////////////
  // Load Site and Site Neighbors
  {
    // reuse persistent site buffers (grow-only)
    site_pitch_in_bytes =
        g_voro.ensure_pitch(g_voro.site_transposed, n_site * sizeof(float), 3);
    site_transposed_dev = (float*)g_voro.site_transposed.p;
    site_knn_pitch_in_bytes =
        g_voro.ensure_pitch(g_voro.site_knn, n_site * sizeof(int), site_k + 1);
    site_knn_dev = (int*)g_voro.site_knn.p;

    site_pitch = site_pitch_in_bytes / sizeof(float);
    site_knn_pitch = site_knn_pitch_in_bytes / sizeof(int);
    // printf("------------------site_pitch: %d, n_site: %d\n", site_pitch,
    //        n_site);
    assert(site_pitch != 0);

    // copy sites to device
    cudaMemcpy2D(site_transposed_dev, site_pitch_in_bytes, site.data(),
                 n_site * sizeof(float), n_site * sizeof(float), 3,
                 cudaMemcpyHostToDevice);
    cuda_check_error();

    // copy site_knn to device, init site_knn as -1
    // site_knn is 2d flat matrix
    // dim: (site_k+1) x n_site
    // each column j store all neighbors of sphere all_medial_spheres.at(j)
    cudaMemcpy2D(site_knn_dev, site_knn_pitch_in_bytes, site_knn.data(),
                 n_site * sizeof(int), n_site * sizeof(int), site_k + 1,
                 cudaMemcpyHostToDevice);
    cuda_check_error();

  }  // Site and Site Neighbors

  //////////////////////////////
  // Store records (stream kept open across calls -- this function runs
  // hundreds of times per pipeline run)
  static std::ofstream record("record.csv", std::ios::app);
  record << "n_site, n_tet, site_k, tet_k, Tet_Sphere, Compute_RPD, "
            "GPU2CPU, Non_Dup_RPCs\n";
  record << std::setprecision(5) << std::setiosflags(std::ios::fixed);

  Stopwatch sw("Iteration");
  double start_time = 0.0, stop_time = 0.0;
  start_time = sw.now();
  record << n_site << ", " << n_tet << ", " << site_k << ", ";

  //////////////////////////////
  // Tet-Sphere
  int tet_k = -1;
  int n_slots_used = 0;  // total (tet, related sphere) pairs = CSR size
  int* tet_knn_csr_dev = nullptr;
  int* slot2tet_dev = nullptr;
  {
    // per-tet counts on device; the relation matrix stays on the GPU.
    // Both big scratch matrices come from the grow-only cache (see
    // compute_tet_sphere_relation) instead of being re-allocated per call.
    const size_t tet_pdist_pitch =
        g_voro.ensure_pitch(g_voro.tet_pdist, n_vert * sizeof(float), n_site) /
        sizeof(float);
    const size_t tet_sphere_relate_pitch =
        g_voro.ensure_pitch(g_voro.tet_relate, n_tet * sizeof(int), n_site) /
        sizeof(int);
    int* tet_sphere_relate_dev = (int*)g_voro.tet_relate.p;
    std::vector<int> tet_counts;
    compute_tet_sphere_relation(
        vert_dev, n_vert, vert_pitch, idx_dev, n_tet, idx_pitch,
        site_transposed_dev, site_flags_dev, n_site, site_pitch,
        site_weights_dev, site_knn_dev, site_k, site_knn_pitch,
        (float*)g_voro.tet_pdist.p, tet_pdist_pitch, tet_sphere_relate_dev,
        tet_sphere_relate_pitch, tet_k, tet_counts, sph_gap);

    // exclusive scan of the per-tet counts -> CSR offsets. Host-side: the
    // counts are already resident here (one small D2H inside the call above)
    // and n_tet ints is nothing next to the buffers this sizes.
    std::vector<int> offsets(n_tet + 1);
    {
      int acc = 0;
      for (int tid = 0; tid < n_tet; ++tid) {
        offsets[tid] = acc;
        acc += tet_counts[tid];
      }
      offsets[n_tet] = acc;
      n_slots_used = acc;
    }
    printf("RPD slots: %d (CSR) vs %lld (dense n_tet*tet_k), %.1f%% of dense\n",
           n_slots_used, (long long)n_tet * tet_k,
           100.0 * n_slots_used / ((double)n_tet * (tet_k > 0 ? tet_k : 1)));

    int* tet_offsets_dev = (int*)g_voro.ensure(g_voro.tet_offsets,
                                               (size_t)(n_tet + 1) * sizeof(int));
    cudaMemcpy(tet_offsets_dev, offsets.data(),
               (size_t)(n_tet + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();
    // n_slots_used can be 0 (no tet relates to any sphere); ensure() with 0
    // bytes would hand back a null pointer, so floor the allocation at one
    // element -- the kernel reads nothing in that case anyway.
    const size_t slot_bytes = (size_t)(n_slots_used > 0 ? n_slots_used : 1) *
                              sizeof(int);
    tet_knn_csr_dev = (int*)g_voro.ensure(g_voro.tet_knn_csr, slot_bytes);
    slot2tet_dev = (int*)g_voro.ensure(g_voro.slot2tet, slot_bytes);
    // fill the CSR directly on device (ascending sid per tet = the exact
    // content, and the exact per-tet order, the dense layout produced)
    fill_tet_knn_csr_dev<<<(n_tet + 255) / 256, 256>>>(
        tet_sphere_relate_dev, tet_sphere_relate_pitch, n_tet, n_site,
        tet_offsets_dev, tet_knn_csr_dev, slot2tet_dev);
    cuda_check_error();
    // tet_sphere_relate_dev belongs to g_voro.tet_relate -- not freed here.
  }
  // // ninwang: debug
  // // copy knn back to host
  // cudaStreamSynchronize(0);
  // tet_knn.resize(n_tet * tet_k);
  // cudaMemcpy2D(tet_knn.data(), n_tet * sizeof(int), tet_knn_dev,
  //              tet_knn_pitch_in_bytes, n_tet * sizeof(int), tet_k,
  //              cudaMemcpyDeviceToHost);
  // cuda_check_error();

  // printf("tet_knn matrix after: \n\t");
  // // for (uint tid = 0; tid < n_tet; tid++) {  // column
  // // for (uint tid = 145; tid < 146; tid++) {  // column
  // uint tid = 33;
  // for (uint i = 0; i < tet_k; i++) {  // row
  //   printf("%d ", tet_knn[i * n_tet + tid]);
  // }
  // printf("\n\t ");
  // // }
  // printf("\n");
  // printf("done compute_tet_weighted_knn_dev \n");

  stop_time = sw.now();  // record Tet_Sphere
  record << tet_k << ", " << stop_time - start_time << ", ";
  start_time = sw.now();

  //////////////////////////////
  // Barycenters
  {
    if (site_is_transposed)
      cudaMemset2D(cell_bary_sum_dev, cell_bary_sum_pitch_in_bytes, 0,
                   n_site * sizeof(float), 3);
    else
      cudaMemset(cell_bary_sum_dev, 0, 3 * n_site * sizeof(float));
    cudaMemset(cell_vol_dev, 0, n_site * sizeof(float));
    cuda_check_error();
  }

  ////////////////////////////////////////////
  // Compute RPD
  //
  // GPU: total thread size = n_grids * n_blocks
  // Each thread compute a ConvexCell defined by
  // one tet and one nearby seed.
  //
  // One thread per CSR slot, i.e. per (tet, related sphere) pair that actually
  // exists -- not per (tet, max-over-all-tets) pair.
  int n_grids = n_slots_used / VORO_BLOCK_SIZE + 1;
  int n_blocks = VORO_BLOCK_SIZE;

  // allocate more, after tet_k been updated
  // by function vcompute_tet_sphere_relation()
  // ninwang: allocate memory for all convex cells (device only -- the host
  // sees just the compacted valid cells, see below)
  ConvexCellTransfer* convex_cells_dev = (ConvexCellTransfer*)g_voro.ensure(
      g_voro.convex_cells,
      (size_t)n_grids * n_blocks * sizeof(ConvexCellTransfer));

  // allocate memory for stats (one per launched thread)
  std::vector<Status> stat((size_t)n_grids * n_blocks,
                           security_radius_not_reached);
  GPUBuffer<Status> gpu_stat(stat);

  {  // GPU voro kernel only
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_check_error();

    // clip tet-cell (init as tet)
    // "voronoi_cells_dev" is not used
    clipped_voro_cell_test_GPU_param_tet<<<n_grids, n_blocks>>>(
        site_transposed_dev, n_site, site_pitch, site_weights_dev,
        site_flags_dev, site_knn_dev, site_knn_pitch, site_k, vert_dev, n_vert,
        vert_pitch, idx_dev, n_tet, idx_pitch, v_adjs_dev, e_adj_offsets_dev,
        e_adj_neighbors_dev, e_adj_vals_dev, f_adjs_dev, f_ids_dev,
        tet_knn_csr_dev, slot2tet_dev, n_slots_used,
        gpu_stat.gpu_data, voronoi_cells_dev, convex_cells_dev,
        cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev, sph_coeffs_dev,
        sph_l_dev, sph_nrm_dev, sh_stride_dev);
    cuda_check_error();

    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    // printf("done clipped_voro_cell_test_GPU_param_tet \n");

    stop_time = sw.now();  // record Compute_RPD
    record << stop_time - start_time << ", ";
    start_time = sw.now();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }  // GPU voro kernel only

  ////////////////////////////////////////////
  // GPU2CPU: copy data back to the cpu -- only the VALID convex cells.
  // flag (device) -> exclusive scan (tiny host scan of #slot ints) -> ordered
  // gather (device) -> single D2H of the compacted cells. Slot order is
  // preserved, so the dedup loop below behaves exactly as it did when it
  // scanned the full buffer.
  int n_valid = 0;
  static std::vector<ConvexCellTransfer> compact_host;  // persistent
  {
    start_time = sw.now();
    const int n_slots = n_grids * n_blocks;
    int* flags_dev =
        (int*)g_voro.ensure(g_voro.cc_flags, (size_t)n_slots * sizeof(int));
    int* pos_dev =
        (int*)g_voro.ensure(g_voro.cc_pos, (size_t)n_slots * sizeof(int));
    flag_valid_cells<<<(n_slots + 255) / 256, 256>>>(convex_cells_dev, n_slots,
                                                     flags_dev);
    cuda_check_error();
    static std::vector<int> flags, pos;  // persistent scratch
    flags.resize(n_slots);
    pos.resize(n_slots);
    cudaMemcpy(flags.data(), flags_dev, (size_t)n_slots * sizeof(int),
               cudaMemcpyDeviceToHost);
    cuda_check_error();
    int acc = 0;
    for (int i = 0; i < n_slots; ++i) {
      pos[i] = acc;
      acc += flags[i];
    }
    n_valid = acc;
    cudaMemcpy2D(site.data(), n_site * sizeof(float), site_transposed_dev,
                 site_pitch_in_bytes, n_site * sizeof(float), 3,
                 cudaMemcpyDeviceToHost);
    if (n_valid > 0) {
      cudaMemcpy(pos_dev, pos.data(), (size_t)n_slots * sizeof(int),
                 cudaMemcpyHostToDevice);
      ConvexCellTransfer* compact_dev = (ConvexCellTransfer*)g_voro.ensure(
          g_voro.cc_compact, (size_t)n_valid * sizeof(ConvexCellTransfer));
      gather_valid_cells<<<(n_slots + 255) / 256, 256>>>(
          convex_cells_dev, flags_dev, pos_dev, n_slots, compact_dev);
      cuda_check_error();
      if (compact_host.size() < (size_t)n_valid) compact_host.resize(n_valid);
      cudaMemcpy(compact_host.data(), compact_dev,
                 (size_t)n_valid * sizeof(ConvexCellTransfer),
                 cudaMemcpyDeviceToHost);
      cuda_check_error();
    }
    stop_time = sw.now();  // gpu2cpu
    record << stop_time - start_time << ", ";
  }  // copy data back to the cpu

  ////////////////////////////////////////////
  // GPU2CPU: copy data back to the cpu
  //
  // stores non-duplicated convex_cells_host
  // seed -> tet_ids, do not process duplicates
  start_time = sw.now();
  std::vector<ConvexCellHost> convex_cells_host_non_dup;
  // flat (tet_id, voro_id) hash dedup replaces map<int,set<int>> — same
  // first-seen-wins over the same iteration order, so the output vector is
  // bit-identical; the node-based containers were the loop's dominant cost.
  std::unordered_set<uint64_t> tetSeedSeen;
  tetSeedSeen.reserve((size_t)n_valid * 2);
  convex_cells_host_non_dup.reserve(n_valid);
  for (int ci = 0; ci < n_valid; ++ci) {
    ConvexCellTransfer& cc_trans = compact_host[ci];
    // this is important!
    // to avoid random value assigned in Status::early_return
    if (!is_convex_cell_valid(cc_trans)) continue;
    // each seed&tet pair should be unique but we might calculate
    // multiple times because of multi-thread, same idea used in
    // get_voro_cell_euler()
    const uint64_t key =
        ((uint64_t)(uint32_t)cc_trans.tet_id << 32) | (uint32_t)cc_trans.voro_id;
    if (!tetSeedSeen.insert(key).second) continue;

    convex_cells_host_non_dup.emplace_back();
    copy_cc(cc_trans, convex_cells_host_non_dup.back());
    // easier for debug
    // assign id for each convex cell as ConvexCellHost::id
    // matching index in convex_cells_host_non_dup
    convex_cells_host_non_dup.back().id = convex_cells_host_non_dup.size() - 1;
  }
  stop_time = sw.now();  // Non_Dup_RPCs
  record << stop_time - start_time << ", ";

  record << std::endl;

  // Device buffers above are persistent (see VoroDevCache) and intentionally
  // not freed here -- they are reused across calls. gpu_stat frees itself via
  // its GPUBuffer destructor.


  return convex_cells_host_non_dup;
}
