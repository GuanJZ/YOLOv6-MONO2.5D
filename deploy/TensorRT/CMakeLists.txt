cmake_minimum_required(VERSION 2.6)

project(yolov6)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/junzhi/softwares/nvidia/TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/include)
link_directories(/home/junzhi/softwares/nvidia/TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov6 ${PROJECT_SOURCE_DIR}/yolov6_mono_2.5d.cpp)
target_link_libraries(yolov6 nvinfer)
target_link_libraries(yolov6 cudart)
target_link_libraries(yolov6 ${OpenCV_LIBS})

add_definitions(-O0 -pthread)