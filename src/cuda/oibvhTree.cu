#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "cuda/oibvhTree.cuh"
#include "cuda/oibvh.cuh"
#include "cuda/utils.cuh"

oibvhTree::oibvhTree(const std::shared_ptr<Mesh> mesh) : m_mesh(mesh), m_convertDone(false), m_buildDone(false)
{
    setup();
}

void oibvhTree::draw(const Shader& shader)
{

    if (!m_convertDone)
    {
        convertToVertexArray();
    }

    glBindVertexArray(m_vertexArrayObj);
    glDrawElements(GL_LINES, static_cast<unsigned int>(m_indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0U);
}

void oibvhTree::convertToVertexArray()
{
    assert(m_buildDone);

    const int primitiveCount = m_faces.size();
    const int numNodesInTree = m_aabbTree.size();
    const int internalNodes = numNodesInTree - primitiveCount;
    const int renderNodes = std::min(internalNodes, 256);

    for (int i = 0; i < renderNodes; i++)
    {
        aabb_box_t aabb = m_aabbTree[i];

        // make cube
        std::vector<glm::vec3> cubeVertices;
        std::vector<unsigned int> cubeIndices;

        make_cube(0.5f * (aabb.maximum.x - aabb.minimum.x),
                  0.5f * (aabb.maximum.y - aabb.minimum.y),
                  0.5 * (aabb.maximum.z - aabb.minimum.z),
                  cubeVertices,
                  cubeIndices);

        int backFaceLowerLeftVertexIndex = 4;
        glm::vec3 backFaceLowerLeftVertex = cubeVertices[backFaceLowerLeftVertexIndex];
        glm::vec3 diff = aabb.minimum - backFaceLowerLeftVertex;

        // shift the bounding box to its real position
        for (int i = 0; i < (int)cubeVertices.size(); ++i)
        {
            glm::vec3 pos = cubeVertices[i] + diff;
            m_vertices.push_back(pos);
        }

        // offset indices
        for (int j = 0; j < (int)cubeIndices.size(); ++j)
        {
            cubeIndices[j] += cubeVertices.size() * i;
        }

        m_indices.insert(m_indices.end(), cubeIndices.begin(), cubeIndices.end());
    }

    glGenVertexArrays(1U, &m_vertexArrayObj);
    glGenBuffers(1U, &m_vertexBufferObj);
    glGenBuffers(1U, &m_elementBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementBufferObj);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0U);
    glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

    glBindVertexArray(0U);

    // convert done
    m_convertDone = true;
}

void oibvhTree::setup()
{
    std::cout << "--set up--" << std::endl;
    for (int i = 0; i < m_mesh->m_facesCount; i++)
    {
        m_faces.push_back(
            glm::uvec3(m_mesh->m_indices[i * 3], m_mesh->m_indices[i * 3 + 1], m_mesh->m_indices[i * 3 + 2]));
    }
    for (auto vertex : m_mesh->m_vertices)
    {
        m_positions.push_back(vertex.m_position);
    }
    std::cout << "faces count: " << m_faces.size() << std::endl;
    std::cout << "vertices count: " << m_positions.size() << std::endl;
}

void oibvhTree::build()
{
    std::cout << "--build oibvh tree--" << std::endl;
    int dev;
    float elapsed_ms;
    cudaGetDevice(&dev);
    std::cout << "device id: " << dev << std::endl;
    const unsigned int primitive_count = m_faces.size();
    const unsigned int vertex_count = m_positions.size();
    // std::cout << oibvh_get_size(2147483647) << std::endl;
    const unsigned int oibvh_size = oibvh_get_size(primitive_count);
    const unsigned int oibvh_internal_node_count = oibvh_size - primitive_count;
    glm::vec3* d_positions;
    glm::uvec3* d_faces;
    aabb_box_t* d_aabbs;
    unsigned int* d_mortons;
    deviceMalloc(&d_positions, vertex_count);
    deviceMalloc(&d_faces, primitive_count);
    deviceMalloc(&d_aabbs, oibvh_size);
    deviceMalloc(&d_mortons, primitive_count);
    deviceMemcpy(d_positions, m_positions.data(), vertex_count);
    deviceMemcpy(d_faces, m_faces.data(), primitive_count);

    elapsed_ms = kernelLaunch([&]() {
        dim3 blockSize = dim3(256);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        calculate_aabb_and_morton_kernel<<<gridSize, blockSize>>>(
            d_faces, d_positions, primitive_count, m_mesh->m_aabb, d_aabbs + oibvh_internal_node_count, d_mortons);
    });
    std::cout << "AABBs and mortons calculation took: " << elapsed_ms << "ms" << std::endl;

#if 0
     // check result
     aabb_box_t* temp_aabbs;
     hostMalloc(&temp_aabbs, primitive_count);
     hostMemcpy(temp_aabbs, d_aabbs + oibvh_internal_node_count, primitive_count);
     aabb_box_t aabb;
     std::cout << sizeof(aabb_box_t) << std::endl;
     aabb.minimum = glm::vec3(1e10);
     aabb.maximum = glm::vec3(-1e10);
     for (int i = 0; i < primitive_count; i++)
    {
        aabb.maximum = glm::max(aabb.maximum, temp_aabbs[i].maximum);
        aabb.minimum = glm::min(aabb.minimum, temp_aabbs[i].minimum);
    }
     aabb == m_mesh->m_aabb ? std::cout << "aabb is correct" << std::endl : std::cout << "aabb is wrong" << std::endl;
     delete[] temp_aabbs;
#endif

    thrust::device_ptr<unsigned int> d_mortons_ptr(d_mortons);
    thrust::device_ptr<glm::uvec3> d_faces_ptr(d_faces);
    thrust::device_ptr<aabb_box_t> d_aabbs_ptr(d_aabbs + oibvh_internal_node_count);
    elapsed_ms = kernelLaunch([&]() {
        thrust::stable_sort_by_key(d_mortons_ptr, d_mortons_ptr + primitive_count, d_faces_ptr);
        thrust::stable_sort_by_key(d_mortons_ptr, d_mortons_ptr + primitive_count, d_aabbs_ptr);
    });
    std::cout << "Sorting took: " << elapsed_ms << "ms" << std::endl;

#if 0
    // print result
    aabb_box_t* temp_aabbs;
    hostMalloc(&temp_aabbs, primitive_count);
    hostMemcpy(temp_aabbs, d_aabbs + oibvh_internal_node_count, primitive_count);
    glm::uvec3* temp_faces;
    hostMalloc(&temp_faces, primitive_count);
    hostMemcpy(temp_faces, d_faces, primitive_count);
    for (int i = 0; i < 100; i++)
    {
        std::cout << temp_aabbs[i].minimum << "," << temp_aabbs[i].maximum << std::endl;
        std::cout << m_positions[temp_faces[i].x] << "," << m_positions[temp_faces[i].y] << "," << m_positions[temp_faces[i].z]
                  << std::endl;
    }
#endif

    const unsigned int primitiveCountNextPower2 = next_power_of_two(primitive_count);
    const unsigned int tHeight = ilog2(primitiveCountNextPower2) + 1;
    const unsigned int tLeafLev = tHeight - 1;
    unsigned int entryLevel = tLeafLev - 1;
    const unsigned int virtualLeafCount = primitiveCountNextPower2 - primitive_count;
    unsigned int entryLevelSize = oibvh_level_real_node_count(entryLevel, tLeafLev, virtualLeafCount);

    std::vector<s_param_t> scheduleParams;
    oibvh_scheduling_parameters(entryLevel, entryLevelSize, THREADS_PER_BLOCK, scheduleParams);

#if 0
    // print result
    std::cout << "scheduleParams: " << std::endl;
    for (auto param : scheduleParams)
    {
        std::cout << param.m_entryLevel << "," << param.m_realCount << "," << param.m_threadsPerGroup << ","
                  << param.m_threads << std::endl;
    }
#endif

    std::cout << "kerenl count: " << scheduleParams.size() << std::endl;

    for (int k = 0; k < scheduleParams.size(); k++)
    {
        std::cout << "kernel" << k << std::endl;
        std::cout << "  entry level: " << scheduleParams[k].m_entryLevel << std::endl;
        std::cout << "  real nodes: " << scheduleParams[k].m_realCount << std::endl;
        std::cout << "  total threads: " << scheduleParams[k].m_threads << std::endl;
        std::cout << "  group size: " << scheduleParams[k].m_threadsPerGroup << std::endl;
        std::cout << "  group count: " << scheduleParams[k].m_threads / scheduleParams[k].m_threadsPerGroup
                  << std::endl;
        elapsed_ms = kernelLaunch([&]() {
            dim3 blockSize = dim3(scheduleParams[k].m_threadsPerGroup);
            dim3 gridSize = dim3(scheduleParams[k].m_threads / scheduleParams[k].m_threadsPerGroup);
            oibvh_tree_construction_kernel<<<gridSize, blockSize>>>(scheduleParams[k].m_entryLevel,
                                                                    scheduleParams[k].m_realCount,
                                                                    primitive_count,
                                                                    scheduleParams[k].m_threadsPerGroup,
                                                                    d_aabbs);
        });
        std::cout << "  oibvh contruct kernel took: " << elapsed_ms << "ms" << std::endl;
    }

#if 0
    // log result
    aabb_box_t* temp_aabbs;
    hostMalloc(&temp_aabbs, oibvh_size);
    hostMemcpy(temp_aabbs, d_aabbs, oibvh_size);
    std::ofstream outfile;
    outfile.open("C://Code//oibvh//logs//bvh_log.txt");
    for (int i = 0; i < oibvh_size; i++)
    {
        outfile << temp_aabbs[i].minimum << "," << temp_aabbs[i].maximum << std::endl;
    }
    //std::cout << m_mesh->m_aabb.minimum << "," << m_mesh->m_aabb.maximum << std::endl;
#endif

    // copy result to host
    m_aabbTree.resize(oibvh_size);
    hostMemcpy(m_aabbTree.data(), d_aabbs, oibvh_size);
    hostMemcpy(m_faces.data(), d_faces, primitive_count);

    // free memory
    cudaFree(d_positions);
    cudaFree(d_faces);
    cudaFree(d_aabbs);
    cudaFree(d_mortons);

    // build done
    m_buildDone = true;
}