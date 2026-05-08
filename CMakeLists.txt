cmake_minimum_required(VERSION 3.10)
project(drone_rt_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 这个就是填写ncnn的配置数的路径，如果不确定在哪，可以使用find / -name "ncnnConfig.cmake" 2>/dev/null 来查找
set(ncnn_DIR "/home/cat/ncnn-install/lib/cmake/ncnn")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

#"drone_rt_detect"就是可执行文件的名称，自己可以改。等编译完(执行完cmake ..和make -j4)再执行./drone_rt_detect 来运行程序（./表示当前目录）
add_executable(Drone_Track
    src/main.cpp
    src/yolo_ncnn.cpp
)

target_link_libraries(Drone_Track
    ${OpenCV_LIBS}
    ncnn
)




# ================= RKNN 单张图片测试程序 =================
#
# 这个 test_rknn_image 是单独的测试程序。
# 它不会影响你原来的 Drone_Track / NCNN 实时检测程序。
#
# 用途：
#   ./test_rknn_image 模型路径 图片路径 输出图片路径

find_path(RKNN_INCLUDE_DIR
    rknn_api.h
    PATHS
       /home/cat/rknpu2/runtime/Linux/librknn_api/include
)

find_library(RKNN_RT_LIB
    rknnrt
    PATHS
       /usr/lib
)

message(STATUS "RKNN_INCLUDE_DIR = ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN_RT_LIB      = ${RKNN_RT_LIB}")

add_executable(test_rknn_image
    src/test_rknn_image.cpp
)

target_include_directories(test_rknn_image PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR}
)

target_link_libraries(test_rknn_image PRIVATE
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
)