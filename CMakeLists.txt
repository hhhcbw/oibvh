cmake_minimum_required (VERSION 3.10)
project (oibvh C CXX CUDA) 

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_INCLUDE_DIRS_PATH" ${CUDA_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MOUDLE_PATH: ${CMAKE_MODULE_PATH}")

find_package(OpenGL REQUIRED)
list(APPEND THIRD_LIBS OpenGL::GL)
find_package(assimp REQUIRED)
list (APPEND THIRD_INCLUDE_DIR ${ASSIMP_INCLUDE_DIRS})
list (APPEND THIRD_LIBS ${ASSIMP_LIBRARIES})
set (oibvh_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (THIRD_BASE_DIR ${oibvh_BASE_DIR}/third)
add_subdirectory(third)
list (APPEND THIRD_INCLUDE_DIR ${THIRD_BASE_DIR} ${GLAD_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${IMGUIGLFWOPENGL3_INCLUDE_DIR})
message (STATUS "third include directory: ${THIRD_INCLUDE_DIR}")
list(APPEND THIRD_LIBS ${GLAD_LIBRARIES} ${GLFW_LIBRARIES} ${IMGUIGLFWOPENGL3_LIBRARIES})

set (oibvh_INCLUDE_DIR ${oibvh_BASE_DIR}/include)
set (oibvh_SRC_DIR ${oibvh_BASE_DIR}/src)

aux_source_directory(${oibvh_SRC_DIR} oibvh_SRC)
aux_source_directory(${oibvh_SRC_DIR}/utils oibvh_SRC)
aux_source_directory(${oibvh_SRC_DIR}/cuda oibvh_SRC)
message(STATUS "oibvh source files: ${oibvh_SRC}")

add_executable(oibvh ${oibvh_SRC} "src/utils/utils.cpp")
target_include_directories(oibvh PUBLIC ${oibvh_INCLUDE_DIR} ${THIRD_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
message(STATUS "third libs: ${THIRD_LIBS}")
target_link_libraries(oibvh PUBLIC ${THIRD_LIBS})