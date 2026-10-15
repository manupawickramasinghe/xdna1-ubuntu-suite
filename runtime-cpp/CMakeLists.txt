# Built from the top-level CMakeLists when XRT is found (or -DXDNA_BUILD_RUNTIME=ON).
find_package(Threads REQUIRED)

add_library(xrt_coreutil SHARED IMPORTED)
set_target_properties(xrt_coreutil PROPERTIES
  IMPORTED_LOCATION "${XRT_ROOT}/lib/libxrt_coreutil.so"
  INTERFACE_INCLUDE_DIRECTORIES "${XRT_ROOT}/include")

add_executable(xdna_daemon xdna_daemon.cpp)
target_include_directories(xdna_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xdna_daemon PRIVATE xrt_coreutil Threads::Threads)
target_compile_options(xdna_daemon PRIVATE -Wall -Wextra)
set_target_properties(xdna_daemon PROPERTIES
  BUILD_RPATH "${XRT_ROOT}/lib"
  INSTALL_RPATH "${XRT_ROOT}/lib")

install(TARGETS xdna_daemon RUNTIME DESTINATION bin)
install(FILES xdna_runner.hpp DESTINATION include/xdna)
