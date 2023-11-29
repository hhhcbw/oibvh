/*********************************************************************
 * @file       model.h
 * @brief      Header file for Model class
 * @details
 * @author     hhhcbw
 * @date       2023-11-21
 *********************************************************************/
#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <string>
#include <vector>

#include "mesh.h"
#include "shader.h"
#include "utils.h"

class Model
{
public:
    Model() = delete;

    /**
     * @brief      Constructor for Model class, expects a filepath to a 3D model.
     * @param[in]  path        File path of model
     * @param[in]  gamma       Have gamma correction for texture or not
     */
    Model(const std::string& path, const bool gamma = false);

    /**
     * @brief       Draws the model, and thus all its meshes.
     * @param[in]   shader    Shader to use
     * @return      void
     */
    void draw(const Shader& shader) const;

private:
    /**
     * @brief      Loads a model with supported ASSIMP extensions from file and
                   stores the resulting meshes in the meshes vector.
     * @param[in]  path       File path of model
     * @return     void
     */
    void loadModel(const std::string& path);

    /**
     * @brief     Processes a node in a recursive fashion
     * @detail    Processes each individual mesh located at the node and repeats this
     *            process on its children nodes (if any).
     * @param[in] node       Node to process
     * @param[in] scene      Whole scene
     * @return    void
     */
    void processNode(const aiNode* node, const aiScene* scene);

    /**
     * @brief       Processes a mesh in node
     * @param[in]   mesh      aiMesh to be process
     * @param[in]   scene     Whole scene
     * @return      Shared pointer of mesh object processed
     */
    std::shared_ptr<Mesh> processMesh(const aiMesh* mesh, const aiScene* scene);

    /**
     * @brief       Checks all material textures of a given type and loads the textures if they're not loaded yet.
     *              The required info is returned as a Texture struct.
     * @param[in]   mat          Material in mesh
     * @param[in]   type         Texture  type
     * @param[in]   typeName     Name of texture type
     * @return      All material textures in mesh
     */
    std::vector<Texture>
    loadMaterialTextures(const aiMaterial* mat, const aiTextureType type, const std::string typeName);

    /**
     * @brief        Load texture from file
     * @param[in]    File path of texture
     * @return       Texture id
     */
    unsigned int textureFromFile(const char* path) const;

public:
    /**
     * @brief Count of vertices in model
     */
    unsigned int m_verticesCount;
    /**
     * @brief Count of faces in model
     */
    unsigned int m_facesCount;
    /**
     * @brief  AABB bounding box of model
     */
    aabb_box_t m_aabb;
    /**
     * @brief  All shared pointer of meshes in model
     */
    std::vector<std::shared_ptr<Mesh>> m_meshes;

private:
    /**
     * @brief  All the textures loaded so far
     * @detail Optimization to make sure textures aren't loaded more than once
     */
    std::vector<Texture> m_texturesLoaded;
    /**
     * @brief  Directory of model file
     */
    std::string m_directory;
    /**
     * @brief  Have gamma correction for texture or not
     */
    bool m_gammaCorrection;
};