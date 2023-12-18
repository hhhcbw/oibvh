#include <device_launch_parameters.h>
#include <device_functions.h>
#include <gProximity/cuda_vectors.h>
#include <gProximity/cuda_intersect_tritri.h>
#include "cuda/collide.cuh"
#include "cuda/oibvh.cuh"
#include "cuda/utils.cuh"

__device__ inline bool overlap(aabb_box_t a, aabb_box_t b)
{
    return (a.minimum.x <= b.maximum.x && a.maximum.x >= b.minimum.x) &&
        (a.minimum.y <= b.maximum.y && a.maximum.y >= b.minimum.y) &&
        (a.minimum.z <= b.maximum.z && a.maximum.z >= b.minimum.z);
}

__device__ inline void read_information(unsigned int* sharedBvhOffsets,
                                        unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedPrimCount,
                                        unsigned int bvhIndex,
                                        unsigned int layoutLength,
                                        unsigned int& bvhOffset,
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
        if (sharedBvhOffsets[m] <= bvhIndex)
        {
            l = m + 1;
            idxLayout = m;
        }
        else
            r = m - 1;
    }
    bvhOffset = sharedBvhOffsets[idxLayout];
    primOffset = sharedPrimOffsets[idxLayout];
    primitiveCount = sharedPrimCount[idxLayout];
#if 0
    printf("\n");
    printf("idxLayout: %d\n", idxLayout);
    printf("bvhOffset: %u\n", bvhOffset);
    printf("primOffset: %u\n", primOffset);
    printf("primitiveCount: %u\n", primitiveCount);
#endif
}

__device__ inline void read_information(unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedVertexOffsets,
                                        unsigned int primIndex,
                                        unsigned int layoutLength,
                                        unsigned int& meshIndex,
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
    meshIndex = idxLayout;
    primOffset = sharedPrimOffsets[idxLayout];
    vertexOffset = sharedVertexOffsets[idxLayout];
}

__global__ void traversal_kernel(bvtt_node_t* src,
                                 bvtt_node_t* dst,
                                 aabb_box_t* aabbs,
                                 tri_pair_node_t* triPairs,
                                 unsigned int* bvhOffsets,
                                 unsigned int* primOffsets,
                                 unsigned int* primCounts,
                                 unsigned int* nextBvttSize,
                                 unsigned int* triPairCount,
                                 unsigned int layoutLength,
                                 unsigned int bvttSize)
{
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIndex = threadIdx.x;
    __shared__ unsigned int sharedBvhOffsets[256];
    __shared__ unsigned int sharedPrimOffsets[256];
    __shared__ unsigned int sharedPrimCounts[256];
    if (localIndex < layoutLength)
    {

        sharedBvhOffsets[localIndex] = bvhOffsets[localIndex];
        sharedPrimOffsets[localIndex] = primOffsets[localIndex];
        sharedPrimCounts[localIndex] = primCounts[localIndex];
#if 0
        printf("\n");
        printf("localIndex: %d\n", localIndex);
        printf("bvhOffset: %u\n", bvhOffsets[localIndex]);
        printf("primOffset: %u\n", primOffsets[localIndex]);
        printf("primitiveCount: %u\n", primCounts[localIndex]);
#endif
    }
    __syncthreads();
    if (globalIndex >= bvttSize)
        return;

    bvtt_node_t node = src[globalIndex];
    const unsigned int bvhIndexA = node.m_bvhIndex[0];
    const unsigned int bvhIndexB = node.m_bvhIndex[1];
    aabb_box_t bvhA = aabbs[bvhIndexA];
    aabb_box_t bvhB = aabbs[bvhIndexB];
    if (!overlap(bvhA, bvhB)) // AABB overlap fail
    {
        return;
    }

    unsigned int bvhOffsetA, primOffsetA, primitiveCountA; // A oibvh tree
    unsigned int bvhOffsetB, primOffsetB, primitiveCountB; // B oibvh tree
    read_information(sharedBvhOffsets,
                     sharedPrimOffsets,
                     sharedPrimCounts,
                     bvhIndexA,
                     layoutLength,
                     bvhOffsetA,
                     primOffsetA,
                     primitiveCountA);
    read_information(sharedBvhOffsets,
                     sharedPrimOffsets,
                     sharedPrimCounts,
                     bvhIndexB,
                     layoutLength,
                     bvhOffsetB,
                     primOffsetB,
                     primitiveCountB);
    const unsigned int primCountNextPower2A = next_power_of_two(primitiveCountA);
    const unsigned int primCountNextPower2B = next_power_of_two(primitiveCountB);
    const unsigned int leafLevA = ilog2(primCountNextPower2A);
    const unsigned int leafLevB = ilog2(primCountNextPower2B);
    const unsigned int realIndexA = bvhIndexA - bvhOffsetA;
    const unsigned int realIndexB = bvhIndexB - bvhOffsetB;
    const unsigned int implicitIndexA = oibvh_real_to_implicit(realIndexA, leafLevA, primitiveCountA);
    const unsigned int implicitIndexB = oibvh_real_to_implicit(realIndexB, leafLevB, primitiveCountB);
#if 0
    if (realIndexA != oibvh_implicit_to_real(implicitIndexA, leafLevA, primitiveCountA))
    {
        printf("implicitIndexA can't map back to realIndexA\n");
    }
    if (realIndexB != oibvh_implicit_to_real(implicitIndexB, leafLevB, primitiveCountB))
    {
        printf("implicitIndexB can't map back to realIndexB\n");
    }
#endif
    const unsigned int levelA = ilog2(implicitIndexA + 1);
    const unsigned int levelB = ilog2(implicitIndexB + 1);
#if 0
    if (levelA >= 1 && levelB >= 1 &&
        !overlap(aabbs[bvhOffsetA + oibvh_implicit_to_real((implicitIndexA - 1) / 2, leafLevA, primitiveCountA)],
                 aabbs[bvhOffsetB + oibvh_implicit_to_real((implicitIndexB - 1) / 2, leafLevB, primitiveCountB)]))
    {
        printf("parent don't overlap!\n");
    }
#endif
#if 0
    printf("\n");
    printf("global index: %u\n", globalIndex);
    printf("bvhOffsetA: %u\n", bvhOffsetA);
    printf("primOffsetA: %u\n", primOffsetA);
    printf("primitiveCountA: %u\n", primitiveCountA);
    printf("primCountNextPower2A: %u\n", primCountNextPower2A);
    printf("levelA: %u\n", levelA);
    printf("leafLevA: %u\n", leafLevA);
    printf("realIndexA: %u\n", realIndexA);
    printf("implicitIndexA: %u\n", implicitIndexA);
    printf("bvhOffsetB: %u\n", bvhOffsetB);
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
        bvttNodeA[bvttCountA++] =
            oibvh_implicit_to_real(implicitIndexA * 2 + 1, leafLevA, primitiveCountA) + bvhOffsetA;
        if (oibvh_have_rchild(implicitIndexA, leafLevA, primitiveCountA))
        {
            // right child
            bvttNodeA[bvttCountA++] =
                oibvh_implicit_to_real(implicitIndexA * 2 + 2, leafLevA, primitiveCountA) + bvhOffsetA;
        }
    }
    else
    {
        // current node
        bvttNodeA[bvttCountA++] = bvhIndexA;
    }
    if (levelB != leafLevB)
    {
        // left child
        bvttNodeB[bvttCountB++] =
            oibvh_implicit_to_real(implicitIndexB * 2 + 1, leafLevB, primitiveCountB) + bvhOffsetB;
        if (oibvh_have_rchild(implicitIndexB, leafLevB, primitiveCountB))
        {
            // right child
            bvttNodeB[bvttCountB++] =
                oibvh_implicit_to_real(implicitIndexB * 2 + 2, leafLevB, primitiveCountB) + bvhOffsetB;
        }
    }
    else
    {
        // current node
        bvttNodeB[bvttCountB++] = bvhIndexB;
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
    unsigned int meshIndexA, meshIndexB, primOffsetA, primOffsetB, vertexOffsetA, vertexOffsetB;
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexA, layoutLength, meshIndexA, primOffsetA, vertexOffsetA);
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexB, layoutLength, meshIndexB, primOffsetB, vertexOffsetB);
    glm::uvec3 triangleA = primitives[triIndexA];
    glm::uvec3 triangleB = primitives[triIndexB];
    float3 triVerticesA[3];
    float3 triVerticesB[3];

#if 0
    printf("\n");
    printf("global index: %u\n", globalIndex);
    printf("triIndexA: %u\n", triIndexA);
    printf("meshIndexA: %u\n", meshIndexA);
    printf("primOffsetA: %u\n", primOffsetA);
    printf("vertexOffsetA: %u\n", vertexOffsetA);
    printf("triIndexB: %u\n", triIndexB);
    printf("meshIndexB: %u\n", meshIndexB);
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
        int_tri_pair_node_t intTriPair{meshIndexA, meshIndexB, triIndexA - primOffsetA, triIndexB - primOffsetB};
        intTriPairs[intTriPairOffset] = intTriPair;
    }
}