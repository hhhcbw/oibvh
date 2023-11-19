cmake_minimum_required (VERSION 3.10)
project (lightPsep)

find_package(OpenGL REQUIRED)
list(APPEND EXTRA_LIBS OpenGL::GL)
set (lightPsep_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (THIRD_BASE_DIR ${lightPsep_BASE_DIR}/third)
add_subdirectory(third)
set (THIRD_INCLUDE_DIR ${GLAD_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${THIRD_BASE_DIR}/glm ${IMGUIGLFWOPENGL3_INCLUDE_DIR})
message (STATUS "third include directory: ${THIRD_INCLUDE_DIR}")
list(APPEND EXTRA_LIBS ${GLAD_LIBRARIES} ${GLFW_LIBRARIES} ${IMGUIGLFWOPENGL3_LIBRARIES})

set (lightPsep_INCLUDE_DIR ${lightPsep_BASE_DIR}/include)
set (lightPsep_SRC_DIR ${lightPsep_BASE_DIR}/src)
set (lightPsep_LIB_DIR ${lightPsep_BASE_DIR}/lib)

aux_source_directory(${lightPsep_SRC_DIR} lightPsep_SRC)
message(STATUS "lightPsep source files: ${lightPsep_SRC}")

add_executable(lightPsep ${lightPsep_SRC})
target_include_directories(lightPsep PUBLIC ${lightPsep_INCLUDE_DIR} ${THIRD_INCLUDE_DIR})
message(STATUS "extra libs: ${EXTRA_LIBS}")
target_link_libraries(lightPsep PUBLIC ${EXTRA_LIBS})
