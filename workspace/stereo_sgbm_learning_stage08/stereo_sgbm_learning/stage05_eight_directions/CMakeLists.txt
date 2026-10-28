cmake_minimum_required(VERSION 3.16)
project(stereo_stage05_eight_directions LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STEREO_INCLUDES
    include
    ../stage04_four_directions/include
    ../stage03_single_path/include
    ../stage02_cost_volume/include
    ../stage01_bm/include)

add_executable(test_sgm_eight_directions tests/test_sgm_eight_directions.cpp)
target_include_directories(test_sgm_eight_directions PRIVATE ${STEREO_INCLUDES})
enable_testing()
add_test(NAME sgm_eight_directions COMMAND test_sgm_eight_directions)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    add_executable(handmade_sgm_8path src/main.cpp)
    target_include_directories(handmade_sgm_8path PRIVATE ${STEREO_INCLUDES})
    target_link_libraries(handmade_sgm_8path PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV not found: building core C++ test only. Install libopencv-dev for image demo.")
endif()
