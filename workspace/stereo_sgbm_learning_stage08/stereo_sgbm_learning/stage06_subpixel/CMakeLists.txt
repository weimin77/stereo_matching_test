cmake_minimum_required(VERSION 3.16)
project(stereo_stage06_subpixel LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STEREO_INCLUDES
    include
    ../stage05_eight_directions/include
    ../stage04_four_directions/include
    ../stage03_single_path/include
    ../stage02_cost_volume/include
    ../stage01_bm/include)

add_executable(test_subpixel_refinement tests/test_subpixel_refinement.cpp)
target_include_directories(test_subpixel_refinement PRIVATE ${STEREO_INCLUDES})
enable_testing()
add_test(NAME subpixel_refinement COMMAND test_subpixel_refinement)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    add_executable(handmade_sgm_subpixel src/main.cpp)
    target_include_directories(handmade_sgm_subpixel PRIVATE ${STEREO_INCLUDES})
    target_link_libraries(handmade_sgm_subpixel PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV not found: core C++ tests built; install libopencv-dev for image demo")
endif()
