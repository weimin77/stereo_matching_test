cmake_minimum_required(VERSION 3.16)
project(stereo_stage02_cost_volume LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 测试只依赖 C++ 标准库，可以单独构建运行。
add_executable(test_cost_volume tests/test_cost_volume.cpp)
target_include_directories(test_cost_volume PRIVATE
    include ../stage01_bm/include)
enable_testing()
add_test(NAME cost_volume_exactness COMMAND test_cost_volume)

# 演示程序用 OpenCV 读取左右图与输出 PNG。
find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    add_executable(handmade_cost_volume src/main.cpp)
    target_include_directories(handmade_cost_volume PRIVATE
        include ../stage01_bm/include)
    target_link_libraries(handmade_cost_volume PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV not found: only the pure C++ test target is built. Install libopencv-dev to build the demo.")
endif()
