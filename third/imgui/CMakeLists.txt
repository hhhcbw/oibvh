add_library(imgui_core STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/imgui.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/imgui_demo.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/imgui_draw.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/imgui_tables.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/imgui_widgets.cpp
	)
target_include_directories(
	imgui_core PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}/backends 
	)
add_library(ImGui::imgui_core ALIAS imgui_core)

add_library(glfw_opengl3 STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_glfw.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
	)
target_include_directories(
	glfw_opengl3 PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/backends 
	${CMAKE_CURRENT_SOURCE_DIR} 
	${GLFW_INCLUDE_DIR}
	)
target_link_libraries(
	glfw_opengl3 PRIVATE 
	ImGui::imgui_core OpenGL::GL
	${GLFW_LIBRARIES}
	)
add_library(ImGui::glfw_opengl3 ALIAS glfw_opengl3)
set (IMGUIGLFWOPENGL3_LIBRARIES ImGui::glfw_opengl3 CACHE INTERNAL "ImGui glfw_opengl3 libraries")
set (IMGUIGLFWOPENGL3_INCLUDE_DIR 	
    ${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}/backends 
    CACHE INTERNAL
    "ImGui glfw_opengl3 include directory")