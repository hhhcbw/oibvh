/*********************************************************************
 * @file       shader.h
 * @brief      Header file for Shader class
 * @details
 * @author     hhhcbw
 * @date       2023-11-20
 *********************************************************************/
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>

class Shader
{
public:
    Shader() = delete;

    /**
     * @brief       Constructor for Shader class
     * @param[in]   vertexPath      File path of vertex shader
     * @param[in]   fragmentPath    File path of fragment shader
     * @param[in]   geometryPath    File path of geometry shader
     */
    Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);

    /**
     * @brief Deconstructor for Shader class
     */
    ~Shader();

    /**
     * @brief  Activate the shader
     * @return void
     */
    void activate() const;

    /**
     * @brief  Deactivate the shader
     * @return void
     */
    void deactivate() const;

#pragma region shaderSetUniform
    /**
     * @brief      Set bool value at uniform location
     * @param[in]  name   Name of uniform variable
     * @param[in]  value  Value to set
     * @return     void
     */
    void setBool(const std::string& name, bool value = true) const;

    /**
     * @brief      Set int value at uniform location
     * @param[in]  name    Name of uniform variable
     * @param[in]  value   Value to set
     * @return     void
     */
    void setInt(const std::string& name, int value) const;

    /**
     * @brief      Set float value at uniform location
     * @param[in]  name      Name of uniform variable
     * @param[in]  value     Value to set
     * @return     void
     */
    void setFloat(const std::string& name, float value) const;

    /**
     * @brief      Set float vec2 value at uniform location
     * @param[in]  name      Name of uniform variable
     * @param[in]  value     Value to set
     * @return     void
     */
    void setVec2(const std::string& name, const glm::vec2& value) const;

    /**
     * @brief      Set float vec2 value at uniform location with two float value
     * @param[in]  name    Name of uniform variable
     * @param[in]  x       First float value
     * @param[in]  y       Second float value
     * @return     void
     */
    void setVec2(const std::string& name, float x, float y) const;

    /**
     * @brief      Set float vec3 value at uniform location
     * @param[in]  name      Name of uniform variable
     * @param[in]  value     Value to set
     * @return     void
     */
    void setVec3(const std::string& name, const glm::vec3& value) const;

    /**
     * @brief      Set float vec3 value at uniform location with three float value
     * @param[in]  name      Name of uniform variable
     * @param[in]  x         First float value
     * @param[in]  y         Second float value
     * @param[in]  z         Third  float value
     * @return     void
     */
    void setVec3(const std::string& name, float x, float y, float z) const;

    /**
     * @brief      Set float vec4 value at uniform location
     * @param[in]  name      Name of uniform variable
     * @param[in]  value     Value to set
     * @return     void
     */
    void setVec4(const std::string& name, const glm::vec4& value) const;

    /**
     * @brief      Set float vec4 value at uniform location with three float value
     * @param[in]  name      Name of uniform variable
     * @param[in]  x         First float value
     * @param[in]  y         Second float value
     * @param[in]  z         Third  float value
     * @param[in]  w         Fourth float value
     * @return     void
     */
    void setVec4(const std::string& name, float x, float y, float z, float w) const;

    /**
     * @brief      Set f32 mat2 value at uniform location
     * @param[in]  name         Name of uniform variable
     * @param[in]  value        Value to set
     * @return     void
     */
    void setMat2(const std::string& name, const glm::mat2& mat) const;

    /**
     * @brief      Set f32 mat3 value at uniform location
     * @param[in]  name         Name of uniform variable
     * @param[in]  value        Value to set
     * @return     void
     */
    void setMat3(const std::string& name, const glm::mat3& mat) const;

    /**
     * @brief      Set f32 mat4 value at uniform location
     * @param[in]  name         Name of uniform variable
     * @param[in]  value        Value to set
     * @return     void
     */
    void setMat4(const std::string& name, const glm::mat4& mat) const;
#pragma endregion shader utility set uniform function declare

private:
    /**
     * @brief      Check shader compilation errors and print if errors occurred
     * @param[in]  shader       Shader to be check
     * @param[in]  type         Type of shader
     * @return     void
     */
    void checkCompileErrors(GLuint shader, std::string type) const;

    /**
     * @brief      Check shader linking errors and print if errors occurred
     * @return     void
     */
    void checkLinkErrors() const;

private:
    /**
     * @brief Id of shader object from glCreateProgram()
     */
    unsigned int m_id;
};