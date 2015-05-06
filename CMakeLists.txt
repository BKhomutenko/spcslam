cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

project( calibration )

find_package( OpenCV REQUIRED )

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

include_directories(include)
add_executable( calibration
    src/main.cpp
)

target_link_libraries( calibration ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( calibration ${CERES_LIBRARIES})

add_executable( cartography_test
    src/cartography.cpp
    src/vision.cpp
    src/matcher.cpp
    src/tests/cartography_tests.cpp
)

target_link_libraries( cartography_test ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( cartography_test ${CERES_LIBRARIES})

add_executable( matching_test
    src/vision.cpp
    src/matcher.cpp
    src/extractor.cpp
    src/tests/matching_tests.cpp
)

target_link_libraries( matching_test ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( matching_test ${CERES_LIBRARIES})

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O2")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()
