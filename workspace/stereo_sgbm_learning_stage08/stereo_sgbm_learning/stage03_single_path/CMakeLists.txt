cmake_minimum_required(VERSION 3.16)
project(stereo_stage03_single_path LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_sgm_single_path tests/test_sgm_single_path.cpp)
target_include_directories(test_sgm_single_path PRIVATE
    include ../stage02_cost_volume/include ../stage01_bm/include)
enable_testing()
add_test(NAME sgm_single_path COMMAND test_sgm_single_path)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    add_executable(handmade_sgm_1path src/main.cpp)
    target_include_directories(handmade_sgm_1path PRIVATE
        include ../stage02_cost_volume/include ../stage01_bm/include)
    target_link_libraries(handmade_sgm_1path PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV not found: building test only. Install libopencv-dev for image demo.")
endif()
