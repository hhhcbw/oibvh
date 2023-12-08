#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include "utils/utils.h"
#include "cuda/utils.cuh"

#define THREADS_PER_BLOCK 256
#define SUBTREESIZE_MAX (THREADS_PER_BLOCK * 2 - 1)

/**
 * @brief       Calculate aabb box and morton code for each face
 * @param[in]   faces          Faces of mesh with three vertices' indices
 * @param[in]   positions      Position of vertices in mesh
 * @param[in]   face_count     Faces count of mesh
 * @param[out]  aabbs          AABB bvh of each face in mesh
 * @return      void
 */
__global__ void
calculate_aabb_kernel(glm::uvec3* faces, glm::vec3* positions, unsigned int face_count, aabb_box_t* aabbs);

/**
 * @brief       Calculate aabb box and morton code for each face
 * @param[in]   faces          Faces of mesh with three vertices' indices
 * @param[in]   positions      Position of vertices in mesh
 * @param[in]   face_count     Faces count of mesh
 * @param[in]   mesh_aabb      AABB bvh of whole mesh
 * @param[out]  aabbs          AABB bvh of each face in mesh
 * @param[out]  mortons        Morton code of each face in mesh
 * @return      void
 */
__global__ void calculate_aabb_and_morton_kernel(glm::uvec3* faces,
                                                 glm::vec3* positions,
                                                 unsigned int face_count,
                                                 aabb_box_t mesh_aabb,
                                                 aabb_box_t* aabbs,
                                                 unsigned int* mortons);

__global__ void oibvh_tree_construction_kernel(unsigned int tEntryLev,
                                               unsigned int realCount,
                                               unsigned int primitive_count,
                                               unsigned int group_size,
                                               aabb_box_t* aabbs);

/**
 * @brief       Calculate size of whole oibvh tree's nodes
 * @param[in]   t         Primitive count
 * @return      Size of all oibvh tree's nodes
 */
__device__ __host__ inline unsigned int oibvh_get_size(const unsigned int t)
{
    return 2 * t - 1 + cuda::std::popcount(next_power_of_two(t) - t);
}

/**
 * @brief       Calculate count of virtual nodes at giving level of oibvh tree
 * @param[in]   li        Giving level of oibvh tree
 * @param[in]   lli       Level of leaf on tree
 * @param[in]   vl        Count of virtual leaf
 * @return      Count of virtual nodes at giving level of oibvh tree
 */
__device__ __host__ inline unsigned int
oibvh_calc_level_virtual_node_count(const unsigned int li, const unsigned int lli, const unsigned int vl)
{
    return vl >> (lli - li);
}

/**
 * @brief       Calculate count of real nodes at giving level of oibvh tree
 * @param[in]   li        Giving level of oibvh tree
 * @param[in]   lli       Level of leaf on tree
 * @param[in]   vl        Count of virtual leaf
 * @return      Count of real nodes at giving level of oibvh tree
 */
__device__ __host__ inline unsigned int
oibvh_level_real_node_count(const unsigned int li, const unsigned int lli, const unsigned int vl)
{
    return (1 << li) - oibvh_calc_level_virtual_node_count(li, lli, vl);
}