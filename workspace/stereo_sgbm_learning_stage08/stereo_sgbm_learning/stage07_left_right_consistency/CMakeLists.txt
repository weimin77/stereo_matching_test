cmake_minimum_required(VERSION 3.16)
project(stereo_stage07_lr_consistency LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STEREO_INCLUDES
    include
    ../stage06_subpixel/include
    ../stage05_eight_directions/include
    ../stage04_four_directions/include
    ../stage03_single_path/include
    ../stage02_cost_volume/include
    ../stage01_bm/include)

add_executable(test_lr_consistency tests/test_lr_consistency.cpp)
target_include_directories(test_lr_consistency PRIVATE ${STEREO_INCLUDES})
enable_testing()
add_test(NAME lr_consistency COMMAND test_lr_consistency)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    add_executable(handmade_sgm_lr src/main.cpp)
    target_include_directories(handmade_sgm_lr PRIVATE ${STEREO_INCLUDES})
    target_link_libraries(handmade_sgm_lr PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV not found: core tests built; install libopencv-dev for image demo")
endif()
