cmake_minimum_required(VERSION 3.14)
project(tracking)

set(CMAKE_CXX_STANDARD 14)
set(TRT_DIR /opt/TensorRT-8.2.3.0)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# CUDA
include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)

# TensorRT
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tracking
        src/strack.cpp
        src/kalman_filter.cpp
        src/byte_tracker.cpp
        src/lapjv.cpp
        src/main.cpp)

target_link_libraries(tracking
        cudart
        nvinfer
        ${OpenCV_LIBS}
        Eigen3::Eigen)

