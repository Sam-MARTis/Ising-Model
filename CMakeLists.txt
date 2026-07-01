cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(CUDAOpenGLInterop LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    BASE_DIR="${CMAKE_SOURCE_DIR}"
)
target_include_directories(${PROJECT_NAME} PRIVATE
    include/   
)



target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    OpenGL::GLU
    glfw
    GLEW::GLEW
    CUDA::cudart
    CUDA::curand
    GLUT::GLUT
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)