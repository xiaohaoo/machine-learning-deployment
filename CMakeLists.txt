
cmake_minimum_required(VERSION 3.8)

# 添加C++11的支持
add_definitions(-std=c++11)

project(machine-learning-deployment VERSION 1.0)

# 设置include路径
include_directories(${PROJECT_SOURCE_DIR}/includes/ ${PROJECT_SOURCE_DIR}/src/)

# 设置动态库的路径
link_directories(${PROJECT_SOURCE_DIR}/libs/)

# 设置源代码的路径
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ SOURCE_LIST)

# 判断操作系统
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(LIB_LIST libonnxruntime.1.10.0 opencv_world455 opencv_world455d)
else()
    set(LIB_LIST libonnxruntime.1.10.0 libopencv_core.405 libopencv_imgcodecs.405 libopencv_videoio.405 libopencv_imgproc.405)
endif(CMAKE_SYSTEM_NAME MATCHES "Windows")

# 定义可执行程序
add_executable(MachineLearningDeployment ${SOURCE_LIST})

# 链接动态库
target_link_libraries (MachineLearningDeployment ${LIB_LIST})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 拷贝文件
file(COPY ${PROJECT_SOURCE_DIR}/assets/img.jpg ${PROJECT_SOURCE_DIR}/resources/onnx.onnx DESTINATION ${PROJECT_BINARY_DIR})

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR})