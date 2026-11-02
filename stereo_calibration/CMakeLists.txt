# Standalone project, or subdirectory of the algorithm_test build. Both work:
#   cmake -S . -B build && cmake --build build          (from this directory)
#   add_subdirectory(stereo_calibration)                (from the parent)
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(stereo_calib LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Same convention as the parent project: prefer the locally built OpenCV 5
    # over the distribution's 4.x install. Override with -DOpenCV_DIR=... .
    set(OPENCV_LOCAL_PREFIX "$ENV{HOME}/opt/opencv5" CACHE PATH "Local OpenCV 5 install prefix")
    if(OPENCV_LOCAL_PREFIX AND IS_DIRECTORY "${OPENCV_LOCAL_PREFIX}/lib/cmake/opencv5")
        list(PREPEND CMAKE_PREFIX_PATH "${OPENCV_LOCAL_PREFIX}")
    endif()
endif()

if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "stereo_calib: OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

add_library(calib_core STATIC calib_core.cpp)
target_include_directories(calib_core PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(calib_core PUBLIC ${OpenCV_LIBS})

add_executable(stereo_calib stereo_calib.cpp)
target_include_directories(stereo_calib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stereo_calib PRIVATE calib_core)

# Qt5 front end. Optional: a machine without the Qt dev packages still builds
# the CLI. The GUI never calls cv::imshow(), so it does not fight the OpenCV
# highgui backend for the event loop.
option(BUILD_STEREO_CALIB_GUI "Build the Qt5 front end of the stereo calibration tool" ON)
if(BUILD_STEREO_CALIB_GUI)
    find_package(Qt5 COMPONENTS Widgets QUIET)
    if(Qt5Widgets_FOUND)
        add_executable(stereo_calib_gui
            stereo_calib_gui.cpp
            gui/main_window.cpp
            gui/calib_worker.cpp
            gui/image_view.cpp
            gui/result_panel.cpp
            gui/params.cpp
            gui/fonts.cpp
        )
        set_target_properties(stereo_calib_gui PROPERTIES AUTOMOC ON)
        target_include_directories(stereo_calib_gui PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/gui
        )
        target_link_libraries(stereo_calib_gui PRIVATE calib_core Qt5::Widgets)
        message(STATUS "stereo_calib: GUI enabled (Qt ${Qt5Widgets_VERSION})")
    else()
        message(WARNING "Qt5 Widgets not found: stereo_calib_gui skipped "
                        "(install qtbase5-dev or configure with -DBUILD_STEREO_CALIB_GUI=OFF)")
    endif()
endif()

# OpenCV 5 moved calibration, rectification and the 3D helpers into separate
# modules, so check for them explicitly instead of letting the compiler report a
# missing symbol later.
set(_missing "")
foreach(module opencv_calib opencv_stereo opencv_geometry opencv_objdetect)
    list(FIND OpenCV_LIB_COMPONENTS ${module} _index)
    if(_index LESS 0)
        list(APPEND _missing ${module})
    endif()
endforeach()
if(_missing)
    # Fatal when configured on its own; a warning when included from the parent
    # build, so a partially built OpenCV does not break the main target too.
    set(_message "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR} lacks ${_missing}, "
                 "which stereo_calib needs. Rebuild OpenCV with those modules enabled.")
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        message(FATAL_ERROR "${_message}")
    else()
        message(WARNING "${_message} Skipping the stereo_calib target.")
        return()
    endif()
endif()
