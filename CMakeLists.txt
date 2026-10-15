cmake_minimum_required(VERSION 3.20)
project(xdna1_ubuntu_suite LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(XDNA_BUILD_RUNTIME AUTO CACHE STRING "Build the XRT runner/daemon/burn tools: AUTO, ON or OFF (needs /opt/xilinx/xrt)")

# Header-only telemetry library
add_library(xdna_telemetry INTERFACE)
target_include_directories(xdna_telemetry INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(xdna_stat tools/xdna_stat.cpp)
target_link_libraries(xdna_stat PRIVATE xdna_telemetry)
target_compile_options(xdna_stat PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS xdna_stat RUNTIME DESTINATION bin)
install(FILES include/xdna_telemetry.hpp include/amdxdna_accel.h DESTINATION include/xdna)

# XRT runtime: built when XRT is present (or forced ON)
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install prefix")
if(XDNA_BUILD_RUNTIME STREQUAL "AUTO")
  if(EXISTS "${XRT_ROOT}/include/xrt/xrt_device.h")
    set(_build_rt ON)
  else()
    set(_build_rt OFF)
    message(STATUS "XRT not found at ${XRT_ROOT}; skipping runtime-cpp (install it with scripts/install_xdna1_ubuntu.sh)")
  endif()
else()
  set(_build_rt ${XDNA_BUILD_RUNTIME})
endif()
if(_build_rt)
  add_subdirectory(runtime-cpp)
endif()
