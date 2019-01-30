cmake_minimum_required(VERSION 3.5.1)
project(java_torch_lib)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH /home/nazar/libtorch)

link_directories(/usr/local/cuda-9.0/lib64)

find_library(curand CURAND)

find_package(Torch REQUIRED)

include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include)
include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include/linux)

set (LIB_NAME ${PROJECT_NAME}${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED jnijavacpp.cpp java_torch_lib.cpp detect_cuda_compute_capabilities.cpp)
target_link_libraries(${LIB_NAME}  "${TORCH_LIBRARIES}")
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 11)


install(TARGETS ${LIB_NAME} LIBRARY DESTINATION .)