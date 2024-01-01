#include <iostream>
#include <fstream>
#include <stack>

#include "cuda/scene.cuh"
#include "cuda/utils.cuh"
#include "cuda/collide.cuh"
#include "cuda/oibvh.cuh"

#define OUTPUT_TIMES 1

Scene::Scene()
    : m_aabbCount(0U), m_primCount(0U), m_vertexCount(0U), m_intTriPairCount(0U), m_detectTimes(0U), m_deviceCount(-1)
{
    deviceMalloc(&m_deviceSrc, 10000000);
    deviceMalloc(&m_deviceDst, 10000000);
    deviceMalloc(&m_deviceTriPairs, 10000000);
    deviceMalloc(&m_deviceAabbs, 10000000);
    deviceMalloc(&m_devicePrims, 10000000);
    deviceMalloc(&m_deviceVertices, 10000000);
    deviceMalloc(&m_deviceIntTriPairs, 10000000);

    glGenVertexArrays(1U, &m_vertexArrayObj);
    glGenBuffers(1U, &m_vertexBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0U);
    glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

    glBindVertexArray(0U);
}

Scene::~Scene()
{
    cudaFree(m_deviceSrc);
    cudaFree(m_deviceDst);
    cudaFree(m_deviceTriPairs);
    cudaFree(m_deviceAabbs);
    cudaFree(m_devicePrims);
    cudaFree(m_deviceVertices);
    cudaFree(m_deviceIntTriPairs);

    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
}

unsigned int Scene::getIntTriPairCount() const
{
    return m_intTriPairCount;
}

void Scene::draw()
{
    if (!m_convertDone)
    {
        convertToVertexArray();
        m_convertDone = true;
    }

    glBindVertexArray(m_vertexArrayObj);
    glDrawArrays(GL_TRIANGLES, 0, m_vertices.size());
    glBindVertexArray(0U);
}

void Scene::convertToVertexArray()
{
    m_vertices.resize(m_intTriPairCount * 6);
    for (int i = 0; i < m_intTriPairCount; i++)
    {
        const auto intTriPair = m_intTriPairs[i];
        const auto bvhA = m_oibvhTrees[intTriPair.m_bvhIndex[0]];
        const auto bvhB = m_oibvhTrees[intTriPair.m_bvhIndex[1]];
        const auto triangleA = bvhA->m_faces[intTriPair.m_triIndex[0]];
        const auto triangleB = bvhB->m_faces[intTriPair.m_triIndex[1]];
        m_vertices[i * 6] = bvhA->m_positions[triangleA.x];
        m_vertices[i * 6 + 1] = bvhA->m_positions[triangleA.y];
        m_vertices[i * 6 + 2] = bvhA->m_positions[triangleA.z];
        m_vertices[i * 6 + 3] = bvhB->m_positions[triangleB.x];
        m_vertices[i * 6 + 4] = bvhB->m_positions[triangleB.y];
        m_vertices[i * 6 + 5] = bvhB->m_positions[triangleB.z];
    }

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(glm::vec3), m_vertices.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_convertDone = true;
}

void Scene::addOibvhTree(std::shared_ptr<OibvhTree> oibvhTree)
{
    assert(oibvhTree->m_buildDone);

    const unsigned int newBvhOffset = m_aabbCount;
    const unsigned int newPrimOffset = m_primCount;
    m_aabbOffsets.push_back(newBvhOffset);
    m_primOffsets.push_back(m_primCount);
    m_vertexOffsets.push_back(m_vertexCount);
    m_primCounts.push_back(oibvhTree->m_faces.size());
    m_aabbCount += oibvhTree->m_aabbTree.size();
    m_primCount += oibvhTree->m_faces.size();
    m_vertexCount += oibvhTree->m_positions.size();
    m_oibvhTrees.push_back(oibvhTree);

#if 0
    std::cout << "--Scene configuration--" << std::endl;
    std::cout << std::endl << "bvh offsets array: ";
    for (const auto& bvhOffset : m_bvhOffsets)
    {
        std::cout << bvhOffset << " ";
    }
    std::cout << std::endl << "primitive offsets array: ";
    for (const auto& primOffset : m_primOffsets)
    {
        std::cout << primOffset << " ";
    }
    std::cout << std::endl << "primitive count array: ";
    for (const auto& primCount : m_primCounts)
    {
        std::cout << primCount << " ";
    }
    std::cout << std::endl;
#endif
}

void Scene::printDeviceInfo(unsigned int deviceId)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    std::cout << "---Print Device Information---" << std::endl;
    std::cout << "Device " << deviceId << ": " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimension: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1]
              << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "Clock rate: " << deviceProp.clockRate / 1000 << "MHz" << std::endl;
    std::cout << "Total constant memory: " << deviceProp.totalConstMem / 1024 << "KB" << std::endl;
    std::cout << "Texture alignment: " << deviceProp.textureAlignment << std::endl;
    std::cout << "Concurrent copy and execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
    std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Kernel execution timeout: " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    std::cout << "Integrated: " << (deviceProp.integrated ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

void Scene::detectCollision(const DeviceType deviceType, const unsigned int entryLevel, const unsigned int expandLevels)
{
    if (m_deviceCount == -1)
    {
        cudaGetDeviceCount(&m_deviceCount);
    }

    if (deviceType == DeviceType::CPU) // collision detection on cpu
    {
        detectCollisionOnCPU();
    }
    else // collision detection on gpu
    {
        assert((int)deviceType < m_deviceCount);
        if (m_detectTimes == 0)
        {
            printDeviceInfo((unsigned int)deviceType);
        }
        detectCollisionOnGPU((unsigned int)deviceType, entryLevel, expandLevels);
    }

    if (m_detectTimes < OUTPUT_TIMES)
    {
        std::cout << std::endl;
    }

    m_detectTimes++;
    m_convertDone = false;
}

void Scene::detectCollisionOnCPU()
{
    if (m_detectTimes < OUTPUT_TIMES)
    {
        std::cout << "---Detecting collision on cpu---" << std::endl;
    }

    // prepare
    std::stack<bvtt_node_t> bvttNodes;
    assert(!m_bvtts.empty());
    for (const auto bvtt : m_bvtts)
    {
        bvttNodes.push(bvtt);
    }

    std::vector<tri_pair_node_t> triPairs;
    // TODO: cpu collision detection
    //
    // while (!bvttNodes.empty())
    //{

    //}
}

void Scene::expandBvttNodes(const unsigned int entryLevel)
{
    m_bvtts.clear();
    for (int i = 0; i < m_aabbOffsets.size(); i++)
        for (int j = i + 1; j < m_aabbOffsets.size(); j++)
        {
            const unsigned int primCountNextPower2A = next_power_of_two(m_primCounts[i]);
            const unsigned int primCountNextPower2B = next_power_of_two(m_primCounts[j]);
            const unsigned int virtualCountA = primCountNextPower2A - m_primCounts[i];
            const unsigned int virtualCountB = primCountNextPower2B - m_primCounts[j];
            const unsigned int leafLevA = ilog2(primCountNextPower2A);
            const unsigned int leafLevB = ilog2(primCountNextPower2B);
            const unsigned int entryLevelA = std::min(leafLevA, entryLevel);
            const unsigned int entryLevelB = std::min(leafLevB, entryLevel);
            const unsigned int startImplicitIdxA = oibvh_get_most_left_descendant_implicitIdx(0, entryLevelA);
            const unsigned int startImplicitIdxB = oibvh_get_most_left_descendant_implicitIdx(0, entryLevelB);
            const unsigned int mostRightValidImplicitIdxA =
                oibvh_get_most_right_valid_implicitIdx(entryLevelA, leafLevA, virtualCountA);
            const unsigned int mostRightValidImplicitIdxB =
                oibvh_get_most_right_valid_implicitIdx(entryLevelB, leafLevB, virtualCountB);

            for (int p = oibvh_implicit_to_real(startImplicitIdxA, leafLevA, virtualCountA);
                 p <= oibvh_implicit_to_real(mostRightValidImplicitIdxA, leafLevA, virtualCountA);
                 p++)
                for (int q = oibvh_implicit_to_real(startImplicitIdxB, leafLevB, virtualCountB);
                     q <= oibvh_implicit_to_real(mostRightValidImplicitIdxB, leafLevB, virtualCountB);
                     q++)
                {
                    m_bvtts.push_back(bvtt_node_t{p + m_aabbOffsets[i], q + m_aabbOffsets[j]});
                }
        }
}

void Scene::detectCollisionOnGPU(const unsigned int deviceId,
                                 const unsigned int entryLevel,
                                 const unsigned int expandLevels)
{
    cudaSetDevice(deviceId);

    if (m_detectTimes < OUTPUT_TIMES)
    {
        std::cout << "---Detecting collision on gpu---" << std::endl;
    }
    // expand initaial bvtt nodes
    expandBvttNodes(entryLevel);

    bool converged = false;
    float elapsed_ms = 0.0f;
    unsigned int h_bvttSize = m_bvtts.size();

    bvtt_node_t* d_src = m_deviceSrc;
    bvtt_node_t* d_dst = m_deviceDst;
    aabb_box_t* d_aabbs = m_deviceAabbs;
    tri_pair_node_t* d_triPairs = m_deviceTriPairs;
    unsigned int* d_aabbOffsets;
    unsigned int* d_primOffsets;
    unsigned int* d_primCounts;
    unsigned int* d_nextBvttSize;
    unsigned int* d_triPairCount;

    // deviceMalloc(&d_src, 2000000);
    // deviceMalloc(&d_dst, 2000000);
    // deviceMalloc(&d_aabbs, m_aabbCount);
    // deviceMalloc(&d_triPairs, 2000000);
    deviceMalloc(&d_aabbOffsets, m_aabbOffsets.size());
    deviceMalloc(&d_primOffsets, m_primOffsets.size());
    deviceMalloc(&d_primCounts, m_primCounts.size());
    deviceMalloc(&d_nextBvttSize, 1);
    deviceMalloc(&d_triPairCount, 1);

    deviceMemcpy(d_src, m_bvtts.data(), m_bvtts.size());
    deviceMemcpy(d_aabbOffsets, m_aabbOffsets.data(), m_aabbOffsets.size());
    deviceMemcpy(d_primOffsets, m_primOffsets.data(), m_primOffsets.size());
    deviceMemcpy(d_primCounts, m_primCounts.data(), m_primCounts.size());
    deviceMemset(d_triPairCount, 0, 1);

    for (int i = 0; i < m_oibvhTrees.size(); i++)
    {
        const auto bvh = m_oibvhTrees[i];
        deviceMemcpy(d_aabbs + m_aabbOffsets[i], bvh->m_aabbTree.data(), bvh->m_aabbTree.size());
    }

    // broad phase
    float elapsed_traversal_ms = 0.0f;
    for (int i = 0; !converged; i++)
    {
        if (m_detectTimes < OUTPUT_TIMES)
        {
            std::cout << "Traversal kernel " << i << ": bvtt size " << h_bvttSize << ", ";
        }
        deviceMemset(d_nextBvttSize, 0, 1);
        elapsed_ms = kernelLaunch([&]() {
            dim3 blockSize =
                dim3(std::min(256U, std::max(next_power_of_two(m_aabbOffsets.size()), next_power_of_two(h_bvttSize))));
            int bx = (h_bvttSize + blockSize.x - 1) / blockSize.x;
            dim3 gridSize = dim3(bx);
            traversal_kernel<<<gridSize, blockSize>>>(d_src,
                                                      d_dst,
                                                      d_aabbs,
                                                      d_triPairs,
                                                      d_aabbOffsets,
                                                      d_primOffsets,
                                                      d_primCounts,
                                                      d_nextBvttSize,
                                                      d_triPairCount,
                                                      m_aabbOffsets.size(),
                                                      h_bvttSize,
                                                      expandLevels);
        });
        if (m_detectTimes < OUTPUT_TIMES)
        {
            std::cout << " traversal took " << elapsed_ms << "ms" << std::endl;
        }
        elapsed_traversal_ms += elapsed_ms;
        hostMemcpy(&h_bvttSize, d_nextBvttSize, 1);
        if (h_bvttSize == 0)
        {
            converged = true;
        }
        else
        {
            std::swap(d_src, d_dst);
        }
    }
    // cudaFree(d_src);
    // cudaFree(d_dst);
    // cudaFree(d_aabbs);
    cudaFree(d_aabbOffsets);
    cudaFree(d_primCounts);
    cudaFree(d_nextBvttSize);

    unsigned int h_triPairCount = 0U;
    hostMemcpy(&h_triPairCount, d_triPairCount, 1);
    if (m_detectTimes % 100 == 0)
    {
        std::cout << "All treversal kernels took: " << elapsed_traversal_ms << "ms" << std::endl;
        std::cout << "Candidate collision triangle pairs count: " << h_triPairCount << std::endl;
    }

#if 0
    // log candidate triangle pairs
    tri_pair_node_t* h_triPairs;
    hostMalloc(&h_triPairs, h_triPairCount);
    hostMemcpy(h_triPairs, d_triPairs, h_triPairCount);
    std::ofstream outfile;
    outfile.open("C://Code//oibvh//logs//candidate_tri_pairs_log.txt");
    for (int i = 0; i < 100; i++)
    {
        outfile << 
    }
#endif

// narrow phase
#if 0
    auto readInfo = [&](const unsigned int triIndx, unsigned int& bvhIndex) {
        int l = 0;
        int r = m_primOffsets.size() - 1;
        int m;
        int layoutIdx;
        while (l <= r)
        {
            m = (l + r) / 2;
            if (triIndx >= m_primOffsets[m])
            {
                l = m + 1;
                layoutIdx = m;
            }
            else
            {
                r = m - 1;
            }
        }
        bvhIndex = layoutIdx;
    };
    tri_pair_node_t* h_triPairs;
    hostMalloc(&h_triPairs, h_triPairCount);
    hostMemcpy(h_triPairs, d_triPairs, h_triPairCount);
    unsigned int intTriPairCount = 0U;
    for (int i = 0; i < h_triPairCount; i++)
    {
        const auto triangleIndexA = h_triPairs[i].m_triIndex[0];
        const auto triangleIndexB = h_triPairs[i].m_triIndex[1];
        unsigned int bvhIndexA, bvhIndexB;
        readInfo(triangleIndexA, bvhIndexA);
        readInfo(triangleIndexB, bvhIndexB);
        const unsigned int triOffsetA = m_primOffsets[bvhIndexA];
        const unsigned int triOffsetB = m_primOffsets[bvhIndexB];
        const auto triangleA = m_oibvhTrees[bvhIndexA]->m_faces[triangleIndexA - triOffsetA];
        const auto triangleB = m_oibvhTrees[bvhIndexB]->m_faces[triangleIndexB - triOffsetB];
        const auto& verticesA = m_oibvhTrees[bvhIndexA]->m_positions;
        const auto& verticesB = m_oibvhTrees[bvhIndexB]->m_positions;
        glm::vec3 P1, P2, P3, Q1, Q2, Q3;
        P1 = verticesA[triangleA.x];
        P2 = verticesA[triangleA.y];
        P3 = verticesA[triangleA.z];
        Q1 = verticesB[triangleB.x];
        Q2 = verticesB[triangleB.y];
        Q3 = verticesB[triangleB.z];
        if (triangleIntersect(P1, P2, P3, Q1, Q2, Q3))
        {
            intTriPairCount++;
        }
    }
    if (m_detectTimes < OUTPUT_TIMES)
    {
        std::cout << "cpu narrow phase intersect triangle pairs count: " << intTriPairCount << std::endl;
    }
    delete[] h_triPairs;
#endif

    glm::uvec3* d_primitives = m_devicePrims;
    glm::vec3* d_vertices = m_deviceVertices;
    int_tri_pair_node_t* d_intTriPairs = m_deviceIntTriPairs;
    unsigned int* d_intTriPairCount; // Count of intersected triangles pair
    unsigned int* d_vertexOffsets;
    // deviceMalloc(&d_primitives, m_primCount);
    // deviceMalloc(&d_vertices, m_vertexCount);
    deviceMalloc(&d_vertexOffsets, m_vertexOffsets.size());
    // deviceMalloc(&d_intTriPairs, 2000000);
    deviceMalloc(&d_intTriPairCount, 1);

    for (int i = 0; i < m_oibvhTrees.size(); i++)
    {
        const auto bvh = m_oibvhTrees[i];
        deviceMemcpy(d_primitives + m_primOffsets[i], bvh->m_faces.data(), bvh->m_faces.size());
        deviceMemcpy(d_vertices + m_vertexOffsets[i], bvh->m_positions.data(), bvh->m_positions.size());
    }
    deviceMemcpy(d_vertexOffsets, m_vertexOffsets.data(), m_vertexOffsets.size());
    deviceMemset(d_intTriPairCount, 0, 1);

    elapsed_ms = kernelLaunch([&]() {
        dim3 blockSize =
            dim3(std::min(256U, std::max(next_power_of_two(m_primOffsets.size()), next_power_of_two(h_triPairCount))));
        int bx = (h_triPairCount + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        triangle_intersect_kernel<<<gridSize, blockSize>>>(d_triPairs,
                                                           d_primitives,
                                                           d_vertices,
                                                           d_primOffsets,
                                                           d_vertexOffsets,
                                                           d_intTriPairs,
                                                           d_intTriPairCount,
                                                           m_primOffsets.size(),
                                                           h_triPairCount);
    });
    hostMemcpy(&m_intTriPairCount, d_intTriPairCount, 1);
    m_intTriPairs.resize(m_intTriPairCount);
    hostMemcpy(m_intTriPairs.data(), d_intTriPairs, m_intTriPairCount);
    if (m_detectTimes % 100 == 0)
    {
        std::cout << "Triangle intersect kernel took: " << elapsed_ms << "ms" << std::endl;
        std::cout << "Intersected triangle pair count: " << m_intTriPairCount << std::endl;
        std::cout << std::endl;
    }

#if 0
    if (m_detectTimes <= OUTPUT_TIMES)
    {
        std::ofstream outfile;
        outfile.open("C://Code//oibvh//logs//int_tri_pair_logs.txt");
        for (int i = 0; i < m_intTriPairCount; i++)
        {
            const auto bvhA = m_objects[m_intTriPairs[i].m_meshIndex[0]].second;
            const auto bvhB = m_objects[m_intTriPairs[i].m_meshIndex[1]].second;
            const auto triA = bvhA->m_faces[m_intTriPairs[i].m_triIndex[0]];
            const auto triB = bvhB->m_faces[m_intTriPairs[i].m_triIndex[1]];
            outfile << bvhA->m_positions[triA.x] << "," << bvhA->m_positions[triA.y] << "," << bvhA->m_positions[triA.z]
                    << " " << bvhB->m_positions[triB.x] << "," << bvhB->m_positions[triB.y] << ","
                    << bvhB->m_positions[triB.z] << std::endl;
        }
    }
#endif

    // cudaFree(d_triPairs);
    cudaFree(d_primOffsets);
    cudaFree(d_triPairCount);
    // cudaFree(d_primitives);
    // cudaFree(d_vertices);
    cudaFree(d_vertexOffsets);
    // cudaFree(d_intTriPairs);
    cudaFree(d_intTriPairCount);
}