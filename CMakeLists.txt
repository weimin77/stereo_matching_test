cmake_minimum_required(VERSION 3.16)

project(algorithm_test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# OpenCV 5.  It is built from source into a local prefix, so nothing is
# registered with the distribution's package manager. Prepend that prefix so
# find_package() prefers it over the system 4.6 install.
#
# Override with:
#   -DOpenCV_DIR=<prefix>/lib/cmake/opencv5    explicit config dir
#   -DOPENCV_LOCAL_PREFIX=<prefix>             different local prefix
#   -DREQUIRE_OPENCV5=OFF                      allow building against 4.x
#
# A CMakeCache.txt left over from a 4.x configure pins OpenCV_DIR to the
# system install; delete the build directory (or the cache) when switching.
# ---------------------------------------------------------------------------
option(REQUIRE_OPENCV5 "Fail if OpenCV 5 is not the version that gets used" ON)
set(OPENCV_LOCAL_PREFIX "$ENV{HOME}/opt/opencv5" CACHE PATH "Local OpenCV 5 install prefix")
if(OPENCV_LOCAL_PREFIX AND IS_DIRECTORY "${OPENCV_LOCAL_PREFIX}/lib/cmake/opencv5")
    list(PREPEND CMAKE_PREFIX_PATH "${OPENCV_LOCAL_PREFIX}")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
if(REQUIRE_OPENCV5 AND OpenCV_VERSION_MAJOR LESS 5)
    message(FATAL_ERROR
        "Found OpenCV ${OpenCV_VERSION}, but OpenCV 5 is required.\n"
        "  - configure with -DOpenCV_DIR=${OPENCV_LOCAL_PREFIX}/lib/cmake/opencv5, or\n"
        "  - delete the build directory if CMakeCache.txt still points at a 4.x install, or\n"
        "  - pass -DREQUIRE_OPENCV5=OFF to build against 4.x on purpose.")
endif()
# Optional: cv::stereo::StereoBinarySGBM lives in opencv_contrib's xstereo module.
# Without it "matcher": "binarysgbm" is compiled out and reports a clear runtime
# error instead of silently falling back to SGBM.
list(FIND OpenCV_LIB_COMPONENTS opencv_xstereo _xstereo_index)
if(_xstereo_index LESS 0)
    message(STATUS "xstereo not found: matcher \"binarysgbm\" disabled (SGBM only)")
else()
    message(STATUS "xstereo found: matcher \"binarysgbm\" available")
    set(ALGORITHM_TEST_HAVE_XSTEREO ON)
endif()

# Report the WLS backend explicitly: it lives in opencv_contrib and is the
# only postprocess backend test.cpp implements. OpenCV_LIB_COMPONENTS lists the
# modules this build actually contains (OpenCV_<mod>_FOUND is only defined for
# components that were explicitly requested from find_package).
list(FIND OpenCV_LIB_COMPONENTS opencv_ximgproc _ximgproc_index)
if(_ximgproc_index LESS 0)
    message(FATAL_ERROR
        "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR} was built without the ximgproc "
        "module, but createDisparityWLSFilterGeneric() is required by test.cpp.\n"
        "  - rebuild OpenCV 5 with -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules, or\n"
        "  - configure with -DOpenCV_DIR=<a build that includes ximgproc>.")
endif()

find_package(OpenMP REQUIRED)
# Optional: nothing in test.cpp / ADCensusBM includes PCL, so a machine
# without libpcl-dev can still build. When PCL is present the include dirs,
# libraries and definitions below are picked up exactly as before.
find_package(PCL QUIET COMPONENTS
    common
    io
    filters
    search
    kdtree
    segmentation
    visualization
)
if(PCL_FOUND)
    message(STATUS "PCL found: ${PCL_DIR}")
else()
    message(STATUS "PCL not found: building without it (no source uses PCL)")
endif()

add_executable(algorithm_test
    test.cpp
    # ADCensusBM core (ADCensus + cross-based aggregation + scanline
    # optimization + refinement). ADCensusBM/main.cpp is intentionally NOT
    # compiled: it needs libconfig++/boost and carries its own main().
    ADCensusBM/adcensuscv.cpp
    ADCensusBM/aggregation.cpp
    ADCensusBM/disparityrefinement.cpp
    ADCensusBM/scanlineoptimization.cpp
    ADCensusBM/stereoprocessor.cpp
)

target_include_directories(algorithm_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ADCensusBM
)

target_link_libraries(algorithm_test PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_definitions(algorithm_test PRIVATE
    ${PCL_DEFINITIONS}
)
if(ALGORITHM_TEST_HAVE_XSTEREO)
    target_compile_definitions(algorithm_test PRIVATE ALGORITHM_TEST_HAVE_XSTEREO=1)
endif()