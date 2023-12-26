#include <device_launch_parameters.h>
#include <device_functions.h>
#include <gProximity/cuda_vectors.h>
#include <gProximity/cuda_intersect_tritri.h>
#include "cuda/collide.cuh"
#include "cuda/oibvh.cuh"
#include "cuda/utils.cuh"

__device__ inline bool overlap(const aabb_box_t& aabb1, const aabb_box_t& aabb2)
{
    return (aabb1.m_minimum.x <= aabb2.m_maximum.x && aabb1.m_maximum.x >= aabb2.m_minimum.x) &&
        (aabb1.m_minimum.y <= aabb2.m_maximum.y && aabb1.m_maximum.y >= aabb2.m_minimum.y) &&
        (aabb1.m_minimum.z <= aabb2.m_maximum.z && aabb1.m_maximum.z >= aabb2.m_minimum.z);
}

__device__ inline void read_information(unsigned int* sharedAabbOffsets,
                                        unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedPrimCount,
                                        unsigned int aabbIndex,
                                        unsigned int layoutLength,
                                        unsigned int& aabbOffset,
                                        unsigned int& primOffset,
                                        unsigned int& primitiveCount)
{
    int l = 0;
    int r = layoutLength - 1;
    int m;
    int idxLayout;
    while (l <= r)
    {
        m = (l + r) / 2;
        if (sharedAabbOffsets[m] <= aabbIndex)
        {
            l = m + 1;
            idxLayout = m;
        }
        else
            r = m - 1;
    }
    aabbOffset = sharedAabbOffsets[idxLayout];
    primOffset = sharedPrimOffsets[idxLayout];
    primitiveCount = sharedPrimCount[idxLayout];
#if 0
    printf("\n");
    printf("idxLayout: %d\n", idxLayout);
    printf("aabbOffset: %u\n", aabbOffset);
    printf("primOffset: %u\n", primOffset);
    printf("primitiveCount: %u\n", primitiveCount);
#endif
}

__device__ inline void read_information(unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedVertexOffsets,
                                        unsigned int primIndex,
                                        unsigned int layoutLength,
                                        unsigned int& bvhIndex,
                                        unsigned int& primOffset,
                                        unsigned int& vertexOffset)
{
    int l = 0;
    int r = layoutLength - 1;
    int m;
    int idxLayout;
    while (l <= r)
    {
        m = (l + r) / 2;
        if (sharedPrimOffsets[m] <= primIndex)
        {
            l = m + 1;
            idxLayout = m;
        }
        else
            r = m - 1;
    }
    bvhIndex = idxLayout;
    primOffset = sharedPrimOffsets[idxLayout];
    vertexOffset = sharedVertexOffsets[idxLayout];
}

__global__ void traversal_kernel(bvtt_node_t* src,
                                 bvtt_node_t* dst,
                                 aabb_box_t* aabbs,
                                 tri_pair_node_t* triPairs,
                                 unsigned int* aabbOffsets,
                                 unsigned int* primOffsets,
                                 unsigned int* primCounts,
                                 unsigned int* nextBvttSize,
                                 unsigned int* triPairCount,
                                 unsigned int layoutLength,
                                 unsigned int bvttSize)
{
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIndex = threadIdx.x;
    __shared__ unsigned int sharedAabbOffsets[256];
    __shared__ unsigned int sharedPrimOffsets[256];
    __shared__ unsigned int sharedPrimCounts[256];
    if (localIndex < layoutLength)
    {

        sharedAabbOffsets[localIndex] = aabbOffsets[localIndex];
        sharedPrimOffsets[localIndex] = primOffsets[localIndex];
        sharedPrimCounts[localIndex] = primCounts[localIndex];
#if 0
        printf("\n");
        printf("localIndex: %d\n", localIndex);
        printf("bvhOffset: %u\n", aabbOffsets[localIndex]);
        printf("primOffset: %u\n", primOffsets[localIndex]);
        printf("primitiveCount: %u\n", primCounts[localIndex]);
#endif
    }
    __syncthreads();
    if (globalIndex >= bvttSize)
        return;

    bvtt_node_t node = src[globalIndex];
    const unsigned int aabbIndexA = node.m_aabbIndex[0];
    const unsigned int aabbIndexB = node.m_aabbIndex[1];
    aabb_box_t aabbA = aabbs[aabbIndexA];
    aabb_box_t aabbB = aabbs[aabbIndexB];
    if (!overlap(aabbA, aabbB)) // AABB overlap fail
    {
        return;
    }

    unsigned int aabbOffsetA, primOffsetA, primitiveCountA; // A oibvh tree
    unsigned int aabbOffsetB, primOffsetB, primitiveCountB; // B oibvh tree
    read_information(sharedAabbOffsets,
                     sharedPrimOffsets,
                     sharedPrimCounts,
                     aabbIndexA,
                     layoutLength,
                     aabbOffsetA,
                     primOffsetA,
                     primitiveCountA);
    read_information(sharedAabbOffsets,
                     sharedPrimOffsets,
                     sharedPrimCounts,
                     aabbIndexB,
                     layoutLength,
                     aabbOffsetB,
                     primOffsetB,
                     primitiveCountB);
    const unsigned int primCountNextPower2A = next_power_of_two(primitiveCountA);
    const unsigned int primCountNextPower2B = next_power_of_two(primitiveCountB);
    const unsigned int virtualCountA = primCountNextPower2A - primitiveCountA;
    const unsigned int virtualCountB = primCountNextPower2B - primitiveCountB;
    const unsigned int leafLevA = ilog2(primCountNextPower2A);
    const unsigned int leafLevB = ilog2(primCountNextPower2B);
    const unsigned int realIndexA = aabbIndexA - aabbOffsetA;
    const unsigned int realIndexB = aabbIndexB - aabbOffsetB;
    const unsigned int implicitIndexA = oibvh_real_to_implicit(realIndexA, leafLevA, virtualCountA);
    const unsigned int implicitIndexB = oibvh_real_to_implicit(realIndexB, leafLevB, virtualCountB);
#if 0
    if (realIndexA != oibvh_implicit_to_real(implicitIndexA, leafLevA, virtualCountA))
    {
        printf("implicitIndexA can't map back to realIndexA\n");
    }
    if (realIndexB != oibvh_implicit_to_real(implicitIndexB, leafLevB, virtualCountB))
    {
        printf("implicitIndexB can't map back to realIndexB\n");
    }
#endif
    const unsigned int levelA = ilog2(implicitIndexA + 1);
    const unsigned int levelB = ilog2(implicitIndexB + 1);
#if 0
    if (levelA >= 1 && levelB >= 1 &&
        !overlap(aabbs[aabbOffsetA + oibvh_implicit_to_real((implicitIndexA - 1) / 2, leafLevA, virtualCountA)],
                 aabbs[aabbOffsetB + oibvh_implicit_to_real((implicitIndexB - 1) / 2, leafLevB, virtualCountB)]))
    {
        printf("parent don't overlap!\n");
    }
#endif
#if 0
    printf("\n");
    printf("global index: %u\n", globalIndex);
    printf("aabbOffsetA: %u\n", aabbOffsetA);
    printf("primOffsetA: %u\n", primOffsetA);
    printf("primitiveCountA: %u\n", primitiveCountA);
    printf("primCountNextPower2A: %u\n", primCountNextPower2A);
    printf("levelA: %u\n", levelA);
    printf("leafLevA: %u\n", leafLevA);
    printf("realIndexA: %u\n", realIndexA);
    printf("implicitIndexA: %u\n", implicitIndexA);
    printf("aabbOffsetB: %u\n", aabbOffsetB);
    printf("primOffsetB: %u\n", primOffsetB);
    printf("primitiveCountB: %u\n", primitiveCountB);
    printf("primCountNextPower2B: %u\n", primCountNextPower2B);
    printf("levelB: %u\n", levelB);
    printf("leafLevB: %u\n", leafLevB);
    printf("realIndexB: %u\n", realIndexB);
    printf("implicitIndexB: %u\n", implicitIndexB);
#endif

    // expand bvtt
    if (levelA == leafLevA && levelB == leafLevB) // a and b are both at leaf node
    {
        const unsigned int primIndexA = primOffsetA + implicitIndexA + 1 - (1 << leafLevA);
        const unsigned int primIndexB = primOffsetB + implicitIndexB + 1 - (1 << leafLevB);
        const tri_pair_node_t triPair{primIndexA, primIndexB};
        const unsigned int triPairIndex = atomicAdd(triPairCount, 1u);
        triPairs[triPairIndex] = triPair;

#if 0
        printf("\n");
        printf("levelA: %u\n", levelA);
        printf("implicitIndexA: %u\n", implicitIndexA);
        printf("primOffsetA: %u\n", primOffsetA);
        printf("primIndexA: %u\n", primIndexA);
        printf("levelB: %u\n", levelB);
        printf("implicitIndexB: %u\n", implicitIndexB);
        printf("primOffsetB: %u\n", primOffsetB);
        printf("primIndexB: %u\n", primIndexB);
#endif

        return;
    }

    unsigned int bvttNodeA[2];
    unsigned int bvttNodeB[2];
    unsigned int bvttCountA = 0;
    unsigned int bvttCountB = 0;
    if (levelA != leafLevA)
    {
        // left child
        bvttNodeA[bvttCountA++] = oibvh_implicit_to_real(implicitIndexA * 2 + 1, leafLevA, virtualCountA) + aabbOffsetA;
        if (oibvh_have_rchild(implicitIndexA, leafLevA, virtualCountA))
        {
            // right child
            bvttNodeA[bvttCountA++] =
                oibvh_implicit_to_real(implicitIndexA * 2 + 2, leafLevA, virtualCountA) + aabbOffsetA;
        }
    }
    else
    {
        // current node
        bvttNodeA[bvttCountA++] = aabbIndexA;
    }
    if (levelB != leafLevB)
    {
        // left child
        bvttNodeB[bvttCountB++] =
            oibvh_implicit_to_real(implicitIndexB * 2 + 1, leafLevB, virtualCountB) + aabbOffsetB;
        if (oibvh_have_rchild(implicitIndexB, leafLevB, virtualCountB))
        {
            // right child
            bvttNodeB[bvttCountB++] =
                oibvh_implicit_to_real(implicitIndexB * 2 + 2, leafLevB, virtualCountB) + aabbOffsetB;
        }
    }
    else
    {
        // current node
        bvttNodeB[bvttCountB++] = aabbIndexB;
    }

    const unsigned int nextBvttOffset = atomicAdd(nextBvttSize, bvttCountA * bvttCountB);
    for (unsigned int i = 0; i < bvttCountA; i++)
        for (unsigned int j = 0; j < bvttCountB; j++)
        {
            dst[nextBvttOffset + i * bvttCountB + j] = bvtt_node_t{bvttNodeA[i], bvttNodeB[j]};
        }
#if 0
    printf("\n");
    printf("global index: %u\n", globalIndex);
    printf("nextBvttOffset: %u\n", nextBvttOffset);
    printf("levelA: %u\n", levelA);
    printf("bvttCountA: %u\n", bvttCountA);
    printf("levelB: %u\n", levelB);
    printf("bvttCountB: %u\n", bvttCountB);
#endif
}

__global__ void triangle_intersect_kernel(tri_pair_node_t* triPairs,
                                          glm::uvec3* primitives,
                                          glm::vec3* vertices,
                                          unsigned int* primOffsets,
                                          unsigned int* vertexOffsets,
                                          int_tri_pair_node_t* intTriPairs,
                                          unsigned int* intTriPairCount,
                                          unsigned int layoutLength,
                                          unsigned int triPairCount)
{
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIndex = threadIdx.x;
    __shared__ unsigned int sharedPrimOffsets[256];
    __shared__ unsigned int sharedVertexOffsets[256];
    if (localIndex < layoutLength)
    {
        sharedPrimOffsets[localIndex] = primOffsets[localIndex];
        sharedVertexOffsets[localIndex] = vertexOffsets[localIndex];
    }
    __syncthreads();
    if (globalIndex >= triPairCount)
        return;

    tri_pair_node_t node = triPairs[globalIndex];
    unsigned int triIndexA = node.m_triIndex[0];
    unsigned int triIndexB = node.m_triIndex[1];
    unsigned int bvhIndexA, bvhIndexB, primOffsetA, primOffsetB, vertexOffsetA, vertexOffsetB;
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexA, layoutLength, bvhIndexA, primOffsetA, vertexOffsetA);
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexB, layoutLength, bvhIndexB, primOffsetB, vertexOffsetB);
    glm::uvec3 triangleA = primitives[triIndexA];
    glm::uvec3 triangleB = primitives[triIndexB];
    float3 triVerticesA[3];
    float3 triVerticesB[3];

#if 0
    printf("\n");
    printf("global index: %u\n", globalIndex);
    printf("triIndexA: %u\n", triIndexA);
    printf("bvhIndexA: %u\n", bvhIndexA);
    printf("primOffsetA: %u\n", primOffsetA);
    printf("vertexOffsetA: %u\n", vertexOffsetA);
    printf("triIndexB: %u\n", triIndexB);
    printf("bvhIndexB: %u\n", bvhIndexB);
    printf("primOffsetB: %u\n", primOffsetB);
    printf("vertexOffsetB: %u\n", vertexOffsetB);
#endif

    for (int i = 0; i < 3; i++)
    {
        glm::vec3 tempVertex = vertices[vertexOffsetA + triangleA[i]];
        triVerticesA[i] = make_float3(tempVertex.x, tempVertex.y, tempVertex.z);
        tempVertex = vertices[vertexOffsetB + triangleB[i]];
        triVerticesB[i] = make_float3(tempVertex.x, tempVertex.y, tempVertex.z);
#if 0
        printf("%f %f %f %f %f %f\n",
               triVerticesA[i].x,
               triVerticesA[i].y,
               triVerticesA[i].z,
               triVerticesB[i].x,
               triVerticesB[i].y,
               triVerticesB[i].z);
#endif
    }

#if 0
    aabb_box_t boxA, boxB;
    boxA.minimum = glm::vec3(1e10);
    boxB.minimum = glm::vec3(1e10);
    boxA.maximum = glm::vec3(1e-10);
    boxB.maximum = glm::vec3(1e-10);
    for (int i = 0; i < 3; i++)
    {
        boxA.minimum = glm::min(boxA.minimum, vertices[vertexOffsetA + triangleA[i]]);
        boxA.maximum = glm::max(boxA.maximum, vertices[vertexOffsetA + triangleA[i]]);
        boxB.minimum = glm::min(boxB.minimum, vertices[vertexOffsetB + triangleB[i]]);
        boxB.maximum = glm::max(boxB.maximum, vertices[vertexOffsetB + triangleB[i]]);
    }
    if (overlap(boxA, boxB))
    {
        atomicAdd(intTriPairCount, 1u);
        //printf("boxA (%f,%f,%f)x(%f,%f,%f) boxB (%f,%f,%f)x(%f,%f,%f)\n",
        //       boxA.minimum.x,
        //       boxA.minimum.y,
        //       boxA.minimum.z,
        //       boxA.maximum.x,
        //       boxA.maximum.y,
        //       boxA.maximum.z,
        //       boxB.minimum.x,
        //       boxB.minimum.y,
        //       boxB.minimum.z,
        //       boxB.maximum.x,
        //       boxB.maximum.y,
        //       boxB.maximum.z);
    }
#endif

    // triangle intersect
    if (triangleIntersection2(
            triVerticesA[0], triVerticesA[1], triVerticesA[2], triVerticesB[0], triVerticesB[1], triVerticesB[2]))
    {
        unsigned int intTriPairOffset = atomicAdd(intTriPairCount, 1u);
        int_tri_pair_node_t intTriPair{bvhIndexA, bvhIndexB, triIndexA - primOffsetA, triIndexB - primOffsetB};
        intTriPairs[intTriPairOffset] = intTriPair;
    }
}