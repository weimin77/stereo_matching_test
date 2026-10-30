cmake_minimum_required(VERSION 3.16)

project(sgbm_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Local OpenCV 5 prefix, same convention as the parent project.
# ---------------------------------------------------------------------------
option(REQUIRE_OPENCV5 "Fail if OpenCV 5 is not the version that gets used" ON)
set(OPENCV_LOCAL_PREFIX "$ENV{HOME}/opt/opencv5" CACHE PATH "Local OpenCV 5 install prefix")
if(OPENCV_LOCAL_PREFIX AND IS_DIRECTORY "${OPENCV_LOCAL_PREFIX}/lib/cmake/opencv5")
    list(PREPEND CMAKE_PREFIX_PATH "${OPENCV_LOCAL_PREFIX}")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
if(REQUIRE_OPENCV5 AND OpenCV_VERSION_MAJOR LESS 5)
    message(FATAL_ERROR "OpenCV ${OpenCV_VERSION} found, but 5.x is required")
endif()

enable_testing()

add_library(sgbm_oracle STATIC tests/oracle/sgbm_oracle.cpp)
target_include_directories(sgbm_oracle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle)
target_link_libraries(sgbm_oracle PUBLIC ${OpenCV_LIBS})
target_compile_options(sgbm_oracle PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(probe_simd tests/probe_simd.cpp)
target_link_libraries(probe_simd PRIVATE sgbm_oracle)
add_test(NAME probe_simd COMMAND probe_simd)

add_executable(debug_case tests/debug_case.cpp)
target_link_libraries(debug_case PRIVATE sgbm_oracle)

add_library(sgbm_repro STATIC
    src/workspace.cpp
    src/prefilter.cpp
    src/cost_volume.cpp
    src/dp_sgbm.cpp
    src/select.cpp
    src/postprocess.cpp
    src/sgbm.cpp
)
target_include_directories(sgbm_repro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sgbm_repro PUBLIC ${OpenCV_LIBS})
target_compile_options(sgbm_repro PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(compare tests/compare.cpp)
target_link_libraries(compare PRIVATE sgbm_repro sgbm_oracle)
add_test(NAME compare COMMAND compare)
set_tests_properties(compare PROPERTIES LABELS agreement)

add_executable(pairs tests/pairs.cpp)
target_link_libraries(pairs PRIVATE sgbm_repro)

add_executable(bench tools/bench.cpp)
target_link_libraries(bench PRIVATE sgbm_repro sgbm_oracle)
