#include <stdio.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include "cuda/oibvhTree.cuh"

__global__ void
calculate_aabbs_kernel(glm::uvec3* faces, glm::vec3* positions, unsigned int face_count, aabb_box_t* aabbs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < face_count)
    {
        glm::uvec3 face = faces[index];
        glm::vec3 v0 = positions[face.x];
        glm::vec3 v1 = positions[face.y];
        glm::vec3 v2 = positions[face.z];
        glm::vec3 minimum = glm::min(glm::min(v0, v1), v2);
        glm::vec3 maximum = glm::max(glm::max(v0, v1), v2);
        aabbs[index].minimum = minimum;
        aabbs[index].maximum = maximum;
    }
}

oibvhTree::oibvhTree(const std::shared_ptr<Mesh> mesh) : m_mesh(mesh)
{
    setup();
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
    cudaGetDevice(&dev);
    std::cout << "device id: " << dev << std::endl;
    unsigned int primitive_count = m_faces.size();
    unsigned int vertex_count = m_positions.size();
    glm::vec3* d_positions;
    glm::uvec3* d_faces;
    aabb_box_t* d_aabb;
    cudaMalloc(&d_positions, vertex_count);
    cudaMalloc(&d_faces, primitive_count);
    cudaMalloc(&d_aabb, primitive_count);
    cudaMemcpy(d_positions, &m_positions[0], vertex_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, &m_faces[0], primitive_count, cudaMemcpyHostToDevice);

    dim3 blockSize = dim3(256);
    int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
    dim3 gridSize = dim3(bx);
    calculate_aabbs_kernel<<<gridSize, blockSize>>>(d_faces, d_positions, primitive_count, d_aabb);
}