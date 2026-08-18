cmake_minimum_required(VERSION 3.16)
set(CMAKE_FIND_FRAMEWORK LAST)
set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)
project(vio_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(vio_cpp src/main.cpp)

target_include_directories(vio_cpp PRIVATE include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(vio_cpp PRIVATE
    ${OpenCV_LIBS}
    Ceres::ceres
    nlohmann_json::nlohmann_json
)
