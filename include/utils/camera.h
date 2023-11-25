/*********************************************************************
 * @file       camera.h
 * @brief      Header file for Camera class
 * @details
 * @author     hhhcbw
 * @date       2023-11-22
 *********************************************************************/
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#define YAW -90.0f
#define PITCH 0.0f
#define SPEED 0.5f
#define SENSITIVITY 0.2f
#define ZOOM 45.0f

/**
 * @brief   Defines several possible options for camera movement
 * @detail  Used as abstraction to stay away from window-system specific input methods
 */
enum CameraMovement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

/**
 * @brief   An abstract camera class that processes input and calculates the corresponding Euler Angles
 * @detail  Vectors and Matrices for use in OpenGL
 */
class Camera
{
public:
    Camera() = delete;

    /**
     * @brief      Constructor for camera class
     * @param[in]  position    Position of camera
     * @param[in]  up          Up vector of camera
     * @param[in]  yaw         Yaw for eluer angles
     * @param[in]  pitch       Pitch for eluer angles
     */
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = YAW,
           float pitch = PITCH);

    /**
     * @brief      Constructor for camera class with scalar value
     * @param[in]  posX        Position.x of camera
     * @param[in]  posY        Position.y of camera
     * @param[in]  posZ        Position.z of camera
     * @param[in]  upX         Up.x of camera
     * @param[in]  upY         Up.y of camera
     * @param[in]  upZ         Up.z of camera
     * @param[in]  yaw         Yaw for eluer angles
     * @param[in]  pitch       Pitch for eluer angles
     */
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

    /**
     * @brief     Calculate view matrix using euler angles and the LookAt matrix
     * @return    View matrix
     */
    glm::mat4 getViewMatrix() const;

    /**
     * @brief     Get zoom of camera
     * @return    Zoom of camera
     */
    float getZoom() const;

    /**
     * @brief      Processes input received from any keyboard-like input system
     * @param[in]  direction       Direction of camera movement
     * @param[in]  deltaTime       Delta time of camera movement
     * @return     void
     */
    void processKeyboard(const CameraMovement direction, const float deltaTime);

    /**
     * @brief       Processes input received from a mouse input system
     * @detail      Expects the offset value in both the x and y direction
     * @param[in]   xoffset            Xoffset to update yaw for eluer angles
     * @param[in]   yoffset            Yoffset to update pitch for eluer angles
     * @param[in]   constrainPitch     Constrain pitch or not
     * @return      void
     */
    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);

    /**
     * @brief     Processes input received from a mouse scroll wheel event
     * @detail    Only requires input on the vertical wheel axis
     * @param[in] yoffset       Yoffset of mouse scroll
     * @return    void
     */
    void processMouseScroll(float yoffset);

private:
    /**
     * @brief   Calculates the front vector from the Camera's (updated) Euler Angles
     * @return  void
     */
    void updateCameraVectors();

private:
    // camera Attributes
    /**
     * @brief Position of the camera
     */
    glm::vec3 m_position;
    /**
     * @brief Front vector of the camera
     */
    glm::vec3 m_front;
    /**
     * @brief Up vector of the camera
     */
    glm::vec3 m_up;
    /**
     * @brief Right vector of the camera
     */
    glm::vec3 m_right;
    /**
     * @brief WorldUp vector of the camera
     */
    glm::vec3 m_worldUp;
    // euler Angles
    /**
     * @brief  Yaw for euler angles
     */
    float m_yaw;
    /**
     * @brief  Pitch for euler angles
     */
    float m_pitch;
    // camera options
    /**
     * @brief  Movement speed of camera
     */
    float m_movementSpeed;
    /**
     * @brief  Sensitivity for mouse
     */
    float m_mouseSensitivity;
    /**
     * @brief  Zoom of camera
     */
    float m_zoom;
};