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

__global__ void oibvh_tree_construction_kernel2(unsigned int tEntryLev,
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
    return 2 * t + cuda::std::popcount(next_power_of_two(t) - t) - 1;
}

/**
 * @brief       Calculate count of virtual nodes at giving level of oibvh tree
 * @param[in]   li        Giving level of oibvh tree
 * @param[in]   lli       Level of leaf on tree
 * @param[in]   vl        Count of virtual leaf
 * @return      Count of virtual nodes at giving level of oibvh tree
 */
__device__ __host__ inline unsigned int
oibvh_level_virtual_node_count(const unsigned int li, const unsigned int lli, const unsigned int vl)
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
    return (1 << li) - oibvh_level_virtual_node_count(li, lli, vl);
}

/**
 * @brief       Calculate count of all virtual nodes at giving level of oibvh tree
 * @param[in]   li        Giving level of oibvh tree
 * @param[in]   lli       Level of leaf on tree
 * @param[in]   vl        Count of virtual leaf
 * @return      Count of all real nodes at giving level of oibvh tree
 */
__device__ __host__ inline unsigned int
oibvh_level_all_virtual_node_count(const unsigned int li, const unsigned int lli, const unsigned int vl)
{
    const unsigned int levelVirtualNodeCount = oibvh_level_virtual_node_count(li, lli, vl);
    return (levelVirtualNodeCount << 1) - cuda::std::popcount(levelVirtualNodeCount);
}

/**
 * @brief        Map implicit index to real index
 * @param[in]    implicitIdx           Implicit  Index
 * @param[in]    leafLev               Level of leaf
 * @param[in]    vl                    Count of virtual leaf
 * @return       Real index mapped from implicit index
 */
__device__ __host__ inline unsigned int
oibvh_implicit_to_real(const unsigned int implicitIdx, const unsigned int leafLev, const unsigned int vl)
{
    const unsigned int level = ilog2(implicitIdx + 1);
    return implicitIdx - oibvh_level_all_virtual_node_count(level - 1, leafLev, vl);
}

/**
 * @brief      Map real index to implicit index
 * @param[in]  realIdx            Real index
 * @param[in]  leafLev            Level of leaf
 * @param[in]  vl                 Count of virtual leaf
 * @return     Implicit index mapped from real index
 */
__device__ __host__ inline unsigned int
oibvh_real_to_implicit(const unsigned int realIdx, const unsigned int leafLev, const unsigned int vl)
{
    if (realIdx == 0)
        return 0;

    const unsigned int level = ilog2(next_power_of_two(realIdx)) - 1;
    const unsigned int levelAllVirtualCount = oibvh_level_all_virtual_node_count(level, leafLev, vl);
    const unsigned int levelAllRealCount = (1 << (level + 1)) - 2 - levelAllVirtualCount;
    if (levelAllRealCount < realIdx)
    {
        return realIdx + levelAllVirtualCount;
    }
    return realIdx + levelAllVirtualCount - oibvh_level_virtual_node_count(level, leafLev, vl);
}

/**
 * @brief       Whether current in oibvh tree have right child
 * @param[in]   implicitIdx        Implicit index of current node
 * @param[in]   leafLev            Level of leaf
 * @param[in]   vl                 Count of virtual leaves
 * @return      True is have right child, otherwise false
 */
__device__ __host__ inline bool
oibvh_have_rchild(const unsigned int implicitIdx, const unsigned int leafLev, const unsigned int vl)
{
    const unsigned int nextLevel = ilog2(implicitIdx + 1) + 1;
#if 0
    printf("%u %u %u %u %u\n",
           nextLevel,
           leafLev,
           primCount,
           2 * implicitIdx + 4 - (1 << nextLevel),
           oibvh_level_real_node_count(nextLevel, leafLev, vl));
#endif
    if (2 * implicitIdx + 4 <= (1 << nextLevel) + oibvh_level_real_node_count(nextLevel, leafLev, vl))
    {
        return true;
    }
    else
    {
        return false;
    }
}