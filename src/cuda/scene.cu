#include <iostream>
#include <fstream>
#include "cuda/scene.cuh"
#include "cuda/utils.cuh"
#include "cuda/collide.cuh"

#define OUTPUT_TIMES 1

Scene::Scene() : m_aabbCount(0U), m_primCount(0U), m_vertexCount(0U), m_intTriPairCount(0U), m_detectTimes(0U)
{
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
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
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
        const auto objectA = m_objects[intTriPair.m_meshIndex[0]];
        const auto objectB = m_objects[intTriPair.m_meshIndex[1]];
        const auto bvhA = objectA.second;
        const auto bvhB = objectB.second;
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

void Scene::addObject(std::pair<std::shared_ptr<Mesh>, std::shared_ptr<OibvhTree>> object)
{
    const auto mesh = object.first;
    const auto bvh = object.second;
    const unsigned int newBvhOffset = m_aabbCount;
    const unsigned int newPrimOffset = m_primCount;
    for (const auto& bvhOffset : m_bvhOffsets)
    {
        m_bvtts.push_back(bvtt_node_t{bvhOffset, newBvhOffset});
    }
    m_bvhOffsets.push_back(newBvhOffset);
    m_primOffsets.push_back(m_primCount);
    m_vertexOffsets.push_back(m_vertexCount);
    m_primCounts.push_back(mesh->m_facesCount);
    m_aabbCount += bvh->m_aabbTree.size();
    m_primCount += mesh->m_facesCount;
    m_vertexCount += mesh->m_verticesCount;
    m_objects.push_back(object);

#if 0
    std::cout << "--Scene configuration--" << std::endl;
    std::cout << "bvtt nodes array: ";
    for (const auto& bvtt : m_bvtts)
    {
        std::cout << "(" << bvtt.m_bvhIndex[0] << "," << bvtt.m_bvhIndex[1] << ") ";
    }
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

void Scene::detectCollision()
{
    m_detectTimes++;
    if (m_detectTimes <= OUTPUT_TIMES)
    {
        std::cout << "--Detecting collision--" << std::endl;
    }
    bool converged = false;
    float elapsed_ms = 0.0f;
    unsigned int h_bvttSize = m_bvtts.size();

    bvtt_node_t* d_src;
    bvtt_node_t* d_dst;
    aabb_box_t* d_aabbs;
    tri_pair_node_t* d_triPairs;
    unsigned int* d_bvhOffsets;
    unsigned int* d_primOffsets;
    unsigned int* d_primCounts;
    unsigned int* d_nextBvttSize;
    unsigned int* d_triPairCount;

    deviceMalloc(&d_src, 10000000);
    deviceMalloc(&d_dst, 10000000);
    deviceMalloc(&d_aabbs, m_aabbCount);
    deviceMalloc(&d_triPairs, 10000000);
    deviceMalloc(&d_bvhOffsets, m_bvhOffsets.size());
    deviceMalloc(&d_primOffsets, m_primOffsets.size());
    deviceMalloc(&d_primCounts, m_primCounts.size());
    deviceMalloc(&d_nextBvttSize, 1);
    deviceMalloc(&d_triPairCount, 1);

    deviceMemcpy(d_src, m_bvtts.data(), m_bvtts.size());
    deviceMemcpy(d_bvhOffsets, m_bvhOffsets.data(), m_bvhOffsets.size());
    deviceMemcpy(d_primOffsets, m_primOffsets.data(), m_primOffsets.size());
    deviceMemcpy(d_primCounts, m_primCounts.data(), m_primCounts.size());
    deviceMemset(d_triPairCount, 0, 1);

    for (int i = 0; i < m_objects.size(); i++)
    {
        const auto bvh = m_objects[i].second;
        deviceMemcpy(d_aabbs + m_bvhOffsets[i], bvh->m_aabbTree.data(), bvh->m_aabbTree.size());
    }

    // broad phase
    float elapsed_traversal_ms = 0.0f;
    for (int i = 0; !converged; i++)
    {
        if (m_detectTimes <= OUTPUT_TIMES)
        {
            std::cout << "Traversal kernel " << i << ": bvtt size " << h_bvttSize << ", ";
        }
        deviceMemset(d_nextBvttSize, 0, 1);
        elapsed_ms = kernelLaunch([&]() {
            dim3 blockSize =
                dim3(std::min(256U, std::max(next_power_of_two(m_bvhOffsets.size()), next_power_of_two(h_bvttSize))));
            int bx = (h_bvttSize + blockSize.x - 1) / blockSize.x;
            dim3 gridSize = dim3(bx);
            traversal_kernel<<<gridSize, blockSize>>>(d_src,
                                                      d_dst,
                                                      d_aabbs,
                                                      d_triPairs,
                                                      d_bvhOffsets,
                                                      d_primOffsets,
                                                      d_primCounts,
                                                      d_nextBvttSize,
                                                      d_triPairCount,
                                                      m_bvhOffsets.size(),
                                                      h_bvttSize);
        });
        if (m_detectTimes <= OUTPUT_TIMES)
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
    if (m_detectTimes <= OUTPUT_TIMES)
    {
        std::cout << "All treversal kernels took: " << elapsed_traversal_ms << "ms" << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_aabbs);
    cudaFree(d_bvhOffsets);
    cudaFree(d_primCounts);
    cudaFree(d_nextBvttSize);

    unsigned int h_triPairCount = 0U;
    hostMemcpy(&h_triPairCount, d_triPairCount, 1);
    if (m_detectTimes <= OUTPUT_TIMES)
    {
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
    glm::uvec3* d_primitives;
    glm::vec3* d_vertices;
    int_tri_pair_node_t* d_intTriPairs;
    unsigned int* d_intTriPairCount; // Count of intersected triangles pair
    unsigned int* d_vertexOffsets;
    deviceMalloc(&d_primitives, m_primCount);
    deviceMalloc(&d_vertices, m_vertexCount);
    deviceMalloc(&d_vertexOffsets, m_vertexOffsets.size());
    deviceMalloc(&d_intTriPairs, 10000000);
    deviceMalloc(&d_intTriPairCount, 1);

    for (int i = 0; i < m_objects.size(); i++)
    {
        const auto bvh = m_objects[i].second;
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
    if (m_detectTimes <= OUTPUT_TIMES)
    {
        std::cout << "Triangle intersect kernel took: " << elapsed_ms << "ms" << std::endl;
        std::cout << "Intersected triangle pair count: " << m_intTriPairCount << std::endl;
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

    cudaFree(d_triPairs);
    cudaFree(d_primOffsets);
    cudaFree(d_triPairCount);
    cudaFree(d_primitives);
    cudaFree(d_vertices);
    cudaFree(d_vertexOffsets);
    cudaFree(d_intTriPairs);
    cudaFree(d_intTriPairCount);

    m_convertDone = false;
}