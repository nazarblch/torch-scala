cmake_minimum_required(VERSION 3.5.1)
project(java_torch_lib)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

set(CMAKE_CXX_STANDARD 11)

# set(CMAKE_PREFIX_PATH /home/nazar/pytorch)
message(STATUS "torch lib path is ${CMAKE_PREFIX_PATH}")


# SET(CMAKE_CXX_COMPILER /usr/bin/g++-5)
# SET(CMAKE_C_COMPILER /usr/bin/gcc-5)
# set(CUDA_HOME /usr/local/cuda-9.0)
# set(CUDAHOSTCXX /usr/bin/g++-5)
# set(CMAKE_C_COMPILER_VERSION 5.5)
# set(CUDA_HOST_COMPILER=/usr/bin/gcc-5)
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)
# set(CUDA_BIN_PATH /usr/local/cuda-9.0/bin)
# set(CUDNN_INCLUDE_DIR /usr/local/cuda-9.0/include)
# set(CUDNN_LIBRARY /usr/local/cuda-9.0/lib64)

find_package(Torch REQUIRED)

message (STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message (STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")

include_directories(${TORCH_INCLUDE_DIRS})

# include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include)
# include_directories(/usr/lib/jvm/java-8-openjdk-amd64/include/linux)

find_package(JNI)

if (${JNI_FOUND})
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

include_directories(${JNI_INCLUDE_DIRS})

#add_executable(main main.cpp)
#target_link_libraries(main  ${TORCH_LIBRARIES})

set (LIB_NAME ${PROJECT_NAME}${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED jnijavacpp.cpp java_torch_lib.cpp)
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${LIB_NAME} ${JNI_LIBRARIES})

#add_executable(test_lib test.cpp)
#target_link_libraries(test_lib ${TORCH_LIBRARIES})
#target_link_libraries(test_lib ${JNI_LIBRARIES})