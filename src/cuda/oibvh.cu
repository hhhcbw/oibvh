#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/extrema.h>
#include "cuda/oibvh.cuh"

__device__ inline unsigned int strech_by_3(unsigned int x)
{
    x = x & 0x3ffu;
    x = (x | (x << 16)) & 0x30000ff;
    x = (x | (x << 8)) & 0x300f00f;
    x = (x | (x << 4)) & 0x30c30c3;
    x = (x | (x << 2)) & 0x9249249;

    return x;
}

__device__ inline unsigned int morton3D(glm::vec3 position)
{
    return (strech_by_3((unsigned int)thrust::min(thrust::max(position.x * 1024.0f, 0.0f), 1023.0f)) << 2 |
            strech_by_3((unsigned int)thrust::min(thrust::max(position.y * 1024.0f, 0.0f), 1023.0f)) << 1 |
            strech_by_3((unsigned int)thrust::min(thrust::max(position.z * 1024.0f, 0.0f), 1023.0f)) << 0);
}

__global__ void
calculate_aabb_kernel(glm::uvec3* faces, glm::vec3* positions, unsigned int face_count, aabb_box_t* aabbs)
{
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index >= face_count)
        return;

    // calculate aabb
    glm::uvec3 face = faces[primitive_index];
    glm::vec3 v0 = positions[face.x];
    glm::vec3 v1 = positions[face.y];
    glm::vec3 v2 = positions[face.z];
    glm::vec3 minimum = glm::min(glm::min(v0, v1), v2);
    glm::vec3 maximum = glm::max(glm::max(v0, v1), v2);
    aabbs[primitive_index].m_minimum = minimum;
    aabbs[primitive_index].m_maximum = maximum;
}

__global__ void calculate_aabb_and_morton_kernel(glm::uvec3* faces,
                                                 glm::vec3* positions,
                                                 unsigned int face_count,
                                                 aabb_box_t mesh_aabb,
                                                 aabb_box_t* aabbs,
                                                 unsigned int* mortons)
{
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index >= face_count)
        return;

    // calculate aabb
    glm::uvec3 face = faces[primitive_index];
    glm::vec3 v0 = positions[face.x];
    glm::vec3 v1 = positions[face.y];
    glm::vec3 v2 = positions[face.z];
    glm::vec3 minimum = glm::min(glm::min(v0, v1), v2);
    glm::vec3 maximum = glm::max(glm::max(v0, v1), v2);
    aabbs[primitive_index].m_minimum = minimum;
    aabbs[primitive_index].m_maximum = maximum;

    // calculate centroid of aabb
    glm::vec3 centroid = (minimum + maximum) * 0.5f;
    glm::vec3 offset = centroid - mesh_aabb.m_minimum;
    glm::vec3 length = mesh_aabb.m_maximum - mesh_aabb.m_minimum;

    // calculate morton code
    mortons[primitive_index] = morton3D(offset / length);
}

__device__ inline aabb_box_t merge_aabb(aabb_box_t leftAABB, aabb_box_t rightAABB)
{
    aabb_box_t mergedAABB;
    mergedAABB.m_minimum = glm::min(leftAABB.m_minimum, rightAABB.m_minimum);
    mergedAABB.m_maximum = glm::max(leftAABB.m_maximum, rightAABB.m_maximum);
    return mergedAABB;
}

__global__ void oibvh_tree_construction_kernel(unsigned int tEntryLev,
                                               unsigned int realCount,
                                               unsigned int primitive_count,
                                               unsigned int group_size,
                                               aabb_box_t* aabbs)
{
    unsigned int tLevPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (tLevPos >= realCount)
        return;
    unsigned int tRealCount = realCount;
    const unsigned int tSubTreeSize = group_size * 2 - 1;
    unsigned int tCacheOffset = (tSubTreeSize >> 1) + threadIdx.x;
    const unsigned int primitiveCountNextPower2 = next_power_of_two(primitive_count);
    const unsigned int tLeafLev = ilog2(primitiveCountNextPower2);
    const unsigned int virtualLeafCount = primitiveCountNextPower2 - primitive_count;
    unsigned int tMostRightRealIdx = oibvh_get_size(tRealCount) - 1;
    unsigned int tMostLeftRealIdx = tMostRightRealIdx - tRealCount + 1;
    unsigned int tRealIdx = tMostLeftRealIdx + tLevPos;
    unsigned int tNextLevelRealCount = oibvh_level_real_node_count(tEntryLev + 1, tLeafLev, virtualLeafCount);
    unsigned int tChildMostRightRealIdx = oibvh_get_size(tNextLevelRealCount) - 1;
    unsigned int tChildMostLeftRealIdx = tChildMostRightRealIdx - tNextLevelRealCount + 1;
    __shared__ aabb_box_t aabbCache[SUBTREESIZE_MAX];
    __shared__ unsigned int askCache[SUBTREESIZE_MAX / 2 + 1];

    unsigned int tChildLeftRealIdx = tChildMostLeftRealIdx + tLevPos * 2;
    unsigned int tChildRightRealIdx = tChildLeftRealIdx + 1;
    aabb_box_t leftAABB = aabbs[tChildLeftRealIdx];
    if (tChildRightRealIdx <= tChildMostRightRealIdx) // right child is real node
    {
        aabbCache[tCacheOffset] = merge_aabb(leftAABB, aabbs[tChildRightRealIdx]);
    }
    else // right child is not real noda
    {
        aabbCache[tCacheOffset] = leftAABB;
    }
    // printf("%u %u %u %u %u %u %u %u %u %u\n",
    //        tLevPos,
    //        tRealCount,
    //        tSubTreeSize,
    //        tCacheOffset,
    //        primitiveCountNextPower2,
    //        tLeafLev,
    //        virtualLeafCount,
    //        tMostRightRealIdx,
    //        tMostLeftRealIdx,
    //        tRealIdx);
    // printf("%u %u %u %u %u\n",
    //       tNextLevelRealCount,
    //       tChildMostRightRealIdx,
    //       tChildMostLeftRealIdx,
    //       tChildLeftRealIdx,
    //       tChildRightRealIdx);
    aabbs[tRealIdx] = aabbCache[tCacheOffset];

    const unsigned int tLevMin = tEntryLev - ilog2(group_size);
    if (tEntryLev == tLevMin)
        return;
    unsigned int tLev = tEntryLev - 1;
    askCache[threadIdx.x] = 0u;
    __syncthreads();
    while (tLev >= tLevMin)
    {
        tCacheOffset = (tCacheOffset - 1) >> 1;
        unsigned int asked = atomicExch(&askCache[tCacheOffset], 1u);
        if (asked == 0u && tLevPos < tRealCount - 1)
            return;
        tLevPos = tLevPos >> 1;
        tChildMostLeftRealIdx = tMostLeftRealIdx;
        tChildMostRightRealIdx = tMostRightRealIdx;
        tRealCount = (tRealCount + 1) >> 1;
        tMostRightRealIdx = tChildMostLeftRealIdx - 1;
        tMostLeftRealIdx = tMostRightRealIdx - tRealCount + 1;

        tChildLeftRealIdx = tChildMostLeftRealIdx + tLevPos * 2;
        tChildRightRealIdx = tChildLeftRealIdx + 1;
        leftAABB = aabbCache[tCacheOffset << 1 | 1];
        if (tChildRightRealIdx <= tChildMostRightRealIdx)
        {
            aabbCache[tCacheOffset] = merge_aabb(leftAABB, aabbCache[tCacheOffset * 2 + 2]);
        }
        else
        {
            aabbCache[tCacheOffset] = leftAABB;
        }
        tRealIdx = tMostLeftRealIdx + tLevPos;
        aabbs[tRealIdx] = aabbCache[tCacheOffset];

        if (tLev == 0)
            return;
        tLev--;
    }
}

__global__ void oibvh_tree_construction_kernel2(unsigned int tEntryLev,
                                                unsigned int realCount,
                                                unsigned int primitive_count,
                                                unsigned int group_size,
                                                aabb_box_t* aabbs)
{
    unsigned int tLevPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (tLevPos >= realCount)
        return;
    unsigned int tRealCount = realCount;
    const unsigned int tSubTreeSize = group_size * 2 - 1;
    unsigned int tCacheOffset = (tSubTreeSize >> 1) + threadIdx.x;
    const unsigned int primitiveCountNextPower2 = next_power_of_two(primitive_count);
    const unsigned int tLeafLev = ilog2(primitiveCountNextPower2);
    const unsigned int virtualLeafCount = primitiveCountNextPower2 - primitive_count;
    unsigned int tImplicitIdx = (1 << ilog2(next_power_of_two(tRealCount))) - 1 + tLevPos;
    unsigned int tRealIdx = oibvh_implicit_to_real(tImplicitIdx, tLeafLev, virtualLeafCount);
#if 0
    if (tImplicitIdx != oibvh_real_to_implicit(tRealIdx, tLeafLev, virtualLeafCount))
    {
        printf("realIdx can't map back to implicitIdx\n");
    }
#endif
    __shared__ aabb_box_t aabbCache[SUBTREESIZE_MAX];
    __shared__ unsigned int askCache[SUBTREESIZE_MAX / 2 + 1];

    unsigned int tChildLeftRealIdx = oibvh_implicit_to_real(tImplicitIdx * 2 + 1, tLeafLev, virtualLeafCount);
    aabb_box_t leftAABB = aabbs[tChildLeftRealIdx];
    if (oibvh_have_rchild(tImplicitIdx, tLeafLev, virtualLeafCount)) // right child is real node
    {
        unsigned int tChildRightRealIdx = tChildLeftRealIdx + 1;
        aabbCache[tCacheOffset] = merge_aabb(leftAABB, aabbs[tChildRightRealIdx]);
    }
    else // right child is not real noda
    {
        aabbCache[tCacheOffset] = leftAABB;
    }
    // printf("%u %u %u %u %u %u %u %u %u %u\n",
    //        tLevPos,
    //        tRealCount,
    //        tSubTreeSize,
    //        tCacheOffset,
    //        primitiveCountNextPower2,
    //        tLeafLev,
    //        virtualLeafCount,
    //        tMostRightRealIdx,
    //        tMostLeftRealIdx,
    //        tRealIdx);
    // printf("%u %u %u %u %u\n",
    //       tNextLevelRealCount,
    //       tChildMostRightRealIdx,
    //       tChildMostLeftRealIdx,
    //       tChildLeftRealIdx,
    //       tChildRightRealIdx);
    aabbs[tRealIdx] = aabbCache[tCacheOffset];

    const unsigned int tLevMin = tEntryLev - ilog2(group_size);
    if (tEntryLev == tLevMin)
        return;
    unsigned int tLev = tEntryLev - 1;
    askCache[threadIdx.x] = 0u;
    __syncthreads();
    while (tLev >= tLevMin)
    {
        tCacheOffset = (tCacheOffset - 1) >> 1;
        unsigned int asked = atomicExch(&askCache[tCacheOffset], 1u);
        if (asked == 0u && tLevPos < tRealCount - 1)
            return;
        tImplicitIdx = (tImplicitIdx - 1) >> 1;
        leftAABB = aabbCache[tCacheOffset << 1 | 1];
        if (oibvh_have_rchild(tImplicitIdx, tLeafLev, virtualLeafCount)) // have right child
        {
            aabbCache[tCacheOffset] = merge_aabb(leftAABB, aabbCache[tCacheOffset * 2 + 2]);
        }
        else // no right child
        {
            aabbCache[tCacheOffset] = leftAABB;
        }
        tRealIdx = oibvh_implicit_to_real(tImplicitIdx, tLeafLev, virtualLeafCount);
        aabbs[tRealIdx] = aabbCache[tCacheOffset];
#if 0
        if (tImplicitIdx != oibvh_real_to_implicit(tRealIdx, tLeafLev, virtualLeafCount))
        {
            printf("realIdx can't map back to implicitIdx\n");
        }
#endif
        if (tLev == 0)
            return;
        tLev--;
    }
}