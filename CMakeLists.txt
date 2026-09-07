cmake_minimum_required(VERSION 3.21)
project(aiter VERSION 1.0.0 LANGUAGES C CXX)

find_package(hip REQUIRED CONFIG HINTS /opt/rocm/lib/cmake/hip)
find_package(Threads REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(aiter SHARED csrc/runtime/runtime.cpp)
target_compile_features(aiter PRIVATE cxx_std_17)
target_include_directories(aiter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(aiter PRIVATE hip::host Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(aiter PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
install(TARGETS aiter EXPORT aiterTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/aiter DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT aiterTargets NAMESPACE aiter:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aiter)
configure_package_config_file(cmake/aiterConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/aiterConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aiter)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/aiterConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/aiterConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/aiterConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aiter)

option(AITER_BUILD_KERNELS "Build native prepared RMSNorm and CK GEMM providers" ON)
set(AITER_GPU_ARCH "gfx950" CACHE STRING "Native provider GPU architecture")
if(AITER_BUILD_KERNELS)
  set(CMAKE_HIP_ARCHITECTURES "${AITER_GPU_ARCH}")
  enable_language(HIP)
  set(AITER_RMS_SOURCES
    csrc/kernels/rmsnorm/rmsnorm_opus_norm_bf16.cu
    csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp16.cu
    csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp32.cu
    csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu)
  set_source_files_properties(${AITER_RMS_SOURCES} PROPERTIES LANGUAGE HIP)
  add_library(aiter_rmsnorm_backend SHARED ${AITER_RMS_SOURCES})
  target_include_directories(aiter_rmsnorm_backend PRIVATE csrc/include)
  target_compile_definitions(aiter_rmsnorm_backend PRIVATE __HIPCC_RTC__)
  set_target_properties(aiter_rmsnorm_backend PROPERTIES HIP_STANDARD 20)
  install(TARGETS aiter_rmsnorm_backend LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/aiter)

  set(AITER_CK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/composable_kernel" CACHE PATH "Pinned CK source directory")
  if(NOT EXISTS "${AITER_CK_DIR}/include/ck/ck.hpp")
    message(FATAL_ERROR "CK headers missing; initialize the pinned submodule or set AITER_BUILD_KERNELS=OFF")
  endif()
  set_source_files_properties(csrc/runtime/ck_blockscale.cu PROPERTIES LANGUAGE HIP)
  add_library(aiter_ck_backend SHARED csrc/runtime/ck_blockscale.cu)
  target_include_directories(aiter_ck_backend PRIVATE csrc/include "${AITER_CK_DIR}/include" "${AITER_CK_DIR}/library/include")
  set_target_properties(aiter_ck_backend PROPERTIES HIP_STANDARD 20)
  install(TARGETS aiter_ck_backend LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/aiter)
endif()

add_executable(aiter_rmsnorm_example examples/native/rmsnorm.cpp)
target_link_libraries(aiter_rmsnorm_example PRIVATE aiter hip::host)

include(CTest)
if(BUILD_TESTING)
  add_executable(aiter_c_api_tests tests/integration/sdk/test_c_api.c)
  target_link_libraries(aiter_c_api_tests PRIVATE aiter)
  add_test(NAME native_c_api COMMAND aiter_c_api_tests)
endif()
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/sdk/test_runtime.cpp")
  add_executable(aiter_native_tests tests/integration/sdk/test_runtime.cpp)
  target_link_libraries(aiter_native_tests PRIVATE aiter hip::host)
endif()
