cmake_minimum_required(VERSION 3.16)
set(CMAKE_FIND_FRAMEWORK LAST)
project(vio_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(vio_cpp src/main.cpp)

target_include_directories(vio_cpp PRIVATE include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(vio_cpp PRIVATE
    ${OpenCV_LIBS}
    Ceres::ceres
)

# ---- AirSim canli entegrasyonu (src/main_airsim.cpp) - OPSIYONEL, bu makinede AIRSIM_ROOT bos oldugu
# icin ATLANIR, vio_cpp (yukaridaki, EuRoC) hedefi ETKILENMEZ. Arkadasinin makinesinde (AirLib kurulu
# oldugu yerde) -DAIRSIM_ROOT=<Cosys-AirSim kok dizini> ile derlenir. Gercek kutuphane dosya adlari/
# uzantilari (libAirLib.a, rpc kutuphanesinin adi) arkadasinin build sistemine gore degisebilir - ilk
# derlemede muhtemelen elle duzeltilecek (bkz. plan Riskler). ----
set(AIRSIM_ROOT "" CACHE PATH "Cosys-AirSim kok dizini (AirLib icerir) - bos ise vio_airsim atlanir")
if(AIRSIM_ROOT)
    add_executable(vio_airsim src/main_airsim.cpp)
    target_include_directories(vio_airsim PRIVATE include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
        ${AIRSIM_ROOT}/AirLib/include
        ${AIRSIM_ROOT}/AirLib/deps/rpclib/include
        ${AIRSIM_ROOT}/AirLib/deps/eigen3)
    target_link_libraries(vio_airsim PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres
        ${AIRSIM_ROOT}/AirLib/lib/libAirLib.a
        rpc)
endif()
