#include <queue>
#include <fstream>

#include "cpu/simpleBVH.h"

SimpleBVH::SimpleBVH(const std::shared_ptr<Mesh> mesh)
    : m_mesh(mesh), m_nodeCount(0U), m_buildDone(false), m_refitDone(true)
{
    std::cout << "---Set up SimpleBVH---" << std::endl;
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
    std::cout << std::endl;
}

inline int next_power2(int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

inline unsigned int log2(int x)
{
    return sizeof(unsigned int) * CHAR_BIT - __lzcnt(x) - 1;
}

void SimpleBVH::build()
{
    std::cout << "---Building SimpleBVH---" << std::endl;

    m_depth = log2(next_power2(m_faces.size()));
    std::cout << "depth of simple bvh tree: " << m_depth << std::endl;
    m_root = recursiveBuild(0, m_faces.size() - 1, 0);

    m_buildDone = true;
    std::cout << "count of nodes in simple bvh: " << m_nodeCount << std::endl;
    std::cout << "aabb bounding box of root: " << m_root->m_aabb << std::endl;

    std::cout << std::endl;
}

void SimpleBVH::log(const std::string& path) const
{
    std::cout << "---Log SimpleBVH---" << std::endl;
    std::queue<std::shared_ptr<simple_bvh_node_t>> q;
    q.push(m_root);
    std::ofstream outfile;
    outfile.open(path.c_str());
    while (!q.empty())
    {
        const auto node = q.front();
        q.pop();
        outfile << node->m_aabb << std::endl;
        if (node->m_left != nullptr)
        {
            q.push(node->m_left);
        }
        if (node->m_right != nullptr)
        {
            q.push(node->m_right);
        }
    }
    std::cout << "log done" << std::endl;
    std::cout << std::endl;
}

void SimpleBVH::refit()
{
    assert(m_buildDone);
    if (!m_refitDone)
    {
        for (int i = 0; i < m_mesh->m_verticesCount; i++)
        {
            m_positions[i] = m_mesh->m_vertices[i].m_position;
        }
        recursiveRefit(m_root);

        m_refitDone = true;
    }
}

void SimpleBVH::recursiveRefit(const std::shared_ptr<simple_bvh_node_t>& node) const
{
    if (node->m_triId != -1) // leaf node
    {
        glm::uvec3 face = m_faces[node->m_triId];
        node->m_aabb.init(m_positions[face[0]], m_positions[face[0]]);
        node->m_aabb.merge(aabb_box_t{m_positions[face[1]], m_positions[face[1]]});
        node->m_aabb.merge(aabb_box_t{m_positions[face[2]], m_positions[face[2]]});
    }
    else // internal node
    {
        recursiveRefit(node->m_left);
        node->m_aabb.init(node->m_left->m_aabb);
        if (node->m_right != nullptr)
        {
            recursiveRefit(node->m_right);
            node->m_aabb.merge(node->m_right->m_aabb);
        }
    }
}

void SimpleBVH::unRefit()
{
    m_refitDone = false;
}

std::shared_ptr<simple_bvh_node_t> SimpleBVH::recursiveBuild(int leftPrim, int rightPrim, int depth)
{
    m_nodeCount++;

    if (leftPrim == rightPrim) // leaf node
    {
        auto node = std::make_shared<simple_bvh_node_t>();
        if (depth < m_depth)
        {
            node->m_left = recursiveBuild(leftPrim, rightPrim, depth + 1);
            node->m_aabb.merge(node->m_left->m_aabb);
            return node;
        }
        glm::uvec3 face = m_faces[leftPrim];
        node->m_aabb.merge(aabb_box_t{m_positions[face[0]], m_positions[face[0]]});
        node->m_aabb.merge(aabb_box_t{m_positions[face[1]], m_positions[face[1]]});
        node->m_aabb.merge(aabb_box_t{m_positions[face[2]], m_positions[face[2]]});
        node->m_triId = leftPrim;
        return node;
    }

    int count = rightPrim - leftPrim + 1;
    int mid = leftPrim + next_power2(count) / 2 - 1;
    assert(mid >= leftPrim);
    assert(mid < rightPrim);

    auto simpleBvhNode = std::make_shared<simple_bvh_node_t>();
    if (count <= (1 << (m_depth - depth - 1)))
    {
        simpleBvhNode->m_left = recursiveBuild(leftPrim, rightPrim, depth + 1);
        simpleBvhNode->m_aabb.merge(simpleBvhNode->m_left->m_aabb);
        return simpleBvhNode;
    }
    simpleBvhNode->m_left = recursiveBuild(leftPrim, mid, depth + 1);
    simpleBvhNode->m_right = recursiveBuild(mid + 1, rightPrim, depth + 1);
    simpleBvhNode->m_aabb.merge(simpleBvhNode->m_left->m_aabb);
    simpleBvhNode->m_aabb.merge(simpleBvhNode->m_right->m_aabb);
    return simpleBvhNode;
}