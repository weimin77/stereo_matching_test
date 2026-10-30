cmake_minimum_required(VERSION 3.16)
project(stereo_rectification LANGUAGES CXX)

# Works both as a standalone project and as an add_subdirectory() of the parent
# build, which already resolved OpenCV.
if(NOT TARGET opencv_calib)
    set(OpenCV_DIR "$ENV{HOME}/opt/opencv5/lib/cmake/opencv5" CACHE PATH
        "OpenCV 5 install prefix (lib/cmake/opencv5)")
    find_package(OpenCV 5 REQUIRED COMPONENTS calib stereo imgproc imgcodecs objdetect)
endif()

set(_missing "")
foreach(module opencv_calib opencv_stereo opencv_imgproc opencv_imgcodecs opencv_objdetect)
    list(FIND OpenCV_LIB_COMPONENTS ${module} _index)
    if(_index LESS 0)
        list(APPEND _missing ${module})
    endif()
endforeach()
if(_missing)
    set(_message "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR} lacks ${_missing}, "
                 "which stereo_rectify needs. Rebuild OpenCV with those modules enabled.")
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        message(FATAL_ERROR "${_message}")
    else()
        message(WARNING "${_message} Skipping the stereo_rectify target.")
        return()
    endif()
endif()

add_executable(stereo_rectify stereo_rectify.cpp)
target_link_libraries(stereo_rectify PRIVATE ${OpenCV_LIBS})
target_compile_features(stereo_rectify PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stereo_rectify PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

message(STATUS "stereo_rectify: OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
