// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context
// creation, etc.) If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>
#include <glad/glad.h>
#include <stb_image.h>
#include <iostream>
#include <utility>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "utils/shader.h"
#include "utils/mesh.h"
#include "utils/model.h"
#include "utils/camera.h"
#include "cuda/oibvhTree.cuh"
#include "cuda/scene.cuh"
#include "cpu/simpleBVH.h"
#include "cpu/simpleCollide.h"

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// camera
Camera camera(glm::vec3(0.0f, 0.7f, 2.5f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool free_view = false;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char**)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.3 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSwapInterval(0); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool draw_bunny1 = false;
    bool draw_box1 = false;
    bool rotate_bunny1 = false;
    bool draw_bunny2 = false;
    bool draw_box2 = false;
    bool rotate_bunny2 = false;
    bool detect_collision = false;
    bool draw_collision = false;
    bool check_result = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    const char vertexIntTriPath[] = "C://Code//oibvh//shaders//temp_vertex_shader.glsl";
    const char fragmentIntTriPath[] = "C://Code//oibvh//shaders//temp_fragment_shader.glsl";
    Shader shaderIntTri(vertexIntTriPath, fragmentIntTriPath);

    const char vertexMeshPath[] = "C://Code//oibvh//shaders//mesh_vertex_shader.glsl";
    const char fragmentMeshPath[] = "C://Code//oibvh//shaders//mesh_fragment_shader.glsl";
    Shader shaderMesh(vertexMeshPath, fragmentMeshPath);

    const char vertexBVHPath[] = "C://Code//oibvh//shaders//bvh_vertex_shader.glsl";
    const char fragmentBVHPath[] = "C://Code//oibvh//shaders//bvh_fragment_shader.glsl";
    Shader shaderBVH(vertexBVHPath, fragmentBVHPath);

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(true);
    Model bunny1("C://Code//oibvh//objects//bunny.obj");
    // Model model("C://Code//oibvh//objects//dragon.obj");
    std::shared_ptr<OibvhTree> treeBunny1 = std::make_shared<OibvhTree>(bunny1.m_meshes[0]);
    // oibvhTree tree(meshSPtr);
    treeBunny1->build();
    Model bunny2(bunny1);
    std::shared_ptr<OibvhTree> treeBunny2 = std::make_shared<OibvhTree>(treeBunny1, bunny2.m_meshes[0]);
    bunny2.m_meshes[0]->translate(glm::vec3(1.0f, 0.0f, 0.0f));
    treeBunny2->refit();
    Scene scene;
    scene.addOibvhTree(treeBunny1);
    scene.addOibvhTree(treeBunny2);

    // simple bvh
    std::shared_ptr<SimpleBVH> simpleBvhBunny1 = std::make_shared<SimpleBVH>(bunny1.m_meshes[0]);
    std::shared_ptr<SimpleBVH> simpleBvhBunny2 = std::make_shared<SimpleBVH>(bunny2.m_meshes[0]);
    simpleBvhBunny1->build();
    // simpleBvhBunny1->log();
    simpleBvhBunny2->build();

    // simple collide
    SimpleCollide simpleCollide;
    simpleCollide.addSimpleBVH(simpleBvhBunny1);
    simpleCollide.addSimpleBVH(simpleBvhBunny2);
    simpleCollide.detect();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            ImGui::Begin("Oibvh collision detect"); // Create a window called "Hello, world!" and append into it.

            ImGui::Checkbox("Draw bunny1", &draw_bunny1);
            ImGui::SameLine();
            ImGui::Checkbox("Draw box1", &draw_box1);
            ImGui::SameLine();
            ImGui::Checkbox("Rotate bunny1", &rotate_bunny1);
            ImGui::Checkbox("Draw bunny2", &draw_bunny2);
            ImGui::SameLine();
            ImGui::Checkbox("Draw box2", &draw_box2);
            ImGui::SameLine();
            ImGui::Checkbox("Rotate bunny2", &rotate_bunny2);
            ImGui::Checkbox("Detect collision", &detect_collision);
            ImGui::SameLine();
            ImGui::Checkbox("Draw collision", &draw_collision);
            ImGui::Checkbox("Check result", &check_result);
            ImGui::Checkbox("Free view", &free_view);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // Rendering
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(
            clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        // glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (draw_bunny1)
        {
            if (rotate_bunny1)
            {
                bunny1.m_meshes[0]->rotateZ();
                treeBunny1->refit();
                simpleBvhBunny1->unRefit();
            }

            shaderMesh.activate();
            // view/projection transformations
            glm::mat4 projection =
                glm::perspective(glm::radians(camera.getZoom()), (float)display_w / (float)display_h, 0.1f, 100.0f);
            glm::mat4 view = camera.getViewMatrix();
            shaderMesh.setMat4("view", view);
            shaderMesh.setMat4("projection", projection);

            // render the loaded model
            glm::mat4 modelMat = glm::mat4(1.0f);
            shaderMesh.setMat4("model", modelMat);
            bunny1.draw(shaderMesh, false);

            shaderMesh.deactivate();

            if (draw_box1)
            {
                glDisable(GL_DEPTH_TEST);
                shaderBVH.activate();
                glm::mat4 modelViewProjection = projection * view * modelMat;
                shaderBVH.setMat4("modelViewProjection", modelViewProjection);
                treeBunny1->draw(shaderBVH);

                shaderBVH.deactivate();
                glEnable(GL_DEPTH_TEST);
            }
        }

        if (draw_bunny2)
        {
            if (rotate_bunny2)
            {
                bunny2.m_meshes[0]->rotateX();
                treeBunny2->refit();
                simpleBvhBunny2->unRefit();
            }

            shaderMesh.activate();
            // view/projection transformations
            glm::mat4 projection =
                glm::perspective(glm::radians(camera.getZoom()), (float)display_w / (float)display_h, 0.1f, 100.0f);
            glm::mat4 view = camera.getViewMatrix();
            shaderMesh.setMat4("view", view);
            shaderMesh.setMat4("projection", projection);

            // render the loaded model
            glm::mat4 modelMat = glm::mat4(1.0f);
            shaderMesh.setMat4("model", modelMat);
            bunny2.draw(shaderMesh, false);

            shaderMesh.deactivate();

            if (draw_box2)
            {
                glDisable(GL_DEPTH_TEST);
                shaderBVH.activate();
                glm::mat4 modelViewProjection = projection * view * modelMat;
                shaderBVH.setMat4("modelViewProjection", modelViewProjection);
                treeBunny2->draw(shaderBVH);

                shaderBVH.deactivate();
                glEnable(GL_DEPTH_TEST);
            }
        }

        if (detect_collision)
        {
            scene.detectCollision();
            // simpleCollide.detect();

            if (draw_collision)
            {
                shaderIntTri.activate();
                // view/projection transformations
                glm::mat4 projection =
                    glm::perspective(glm::radians(camera.getZoom()), (float)display_w / (float)display_h, 0.1f, 100.0f);
                glm::mat4 view = camera.getViewMatrix();
                shaderIntTri.setMat4("view", view);
                shaderIntTri.setMat4("projection", projection);
                glm::mat4 modelMat = glm::mat4(1.0f);
                shaderIntTri.setMat4("model", modelMat);

                scene.draw();
                // simpleCollide.draw();

                shaderIntTri.deactivate();
            }
        }

        if (check_result)
        {
            simpleBvhBunny1->refit();
            simpleBvhBunny2->refit();
            simpleCollide.detect();
            if (!simpleCollide.check(scene.getIntTriPairCount()))
            {
                // wrong count of intersect triangle pairs
                std::cout << "ERROR: Wrong count of intersect triangle pairs!!!" << std::endl;
                std::cout << "Expected: " << simpleCollide.getIntTriPairCount() << std::endl;
                std::cout << "Actual: " << scene.getIntTriPairCount() << std::endl;
                exit(1);
            }
        }

        free_view ? glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED)
                  : glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        free_view = !free_view;
    }

    if (free_view)
    { // control camera
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.processKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.processKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.processKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.processKeyboard(RIGHT, deltaTime);
    }
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (!free_view)
    {
        return;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (free_view)
    {
        camera.processMouseScroll(static_cast<float>(yoffset));
    }
}