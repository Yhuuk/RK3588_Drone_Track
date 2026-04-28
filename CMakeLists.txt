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