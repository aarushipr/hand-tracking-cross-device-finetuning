cmake_minimum_required(VERSION 3.10...3.18)

find_package(GTest CONFIG REQUIRED)

add_library(basalt-headers-test-utils INTERFACE)
add_library (basalt::basalt-headers-test-utils ALIAS basalt-headers-test-utils)
target_link_libraries(basalt-headers-test-utils INTERFACE GTest::gtest)

target_include_directories(basalt-headers-test-utils INTERFACE include)

target_compile_definitions(basalt-headers-test-utils INTERFACE BASALT_ENABLE_BOUNDS_CHECKS)

# List all header files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
target_sources(basalt-headers-test-utils
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/test_utils.h
)

add_executable(test_image src/test_image.cpp)
target_link_libraries(test_image GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_spline_rd src/test_spline_rd.cpp)
target_link_libraries(test_spline_rd GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_spline_so3 src/test_spline_so3.cpp)
target_link_libraries(test_spline_so3 GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_bezier_rd src/test_bezier_rd.cpp)
target_link_libraries(test_bezier_rd GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_spline_se3 src/test_spline_se3.cpp)
target_link_libraries(test_spline_se3 GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_camera src/test_camera.cpp)
target_link_libraries(test_camera GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_sophus src/test_sophus.cpp)
target_link_libraries(test_sophus GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_preintegration src/test_preintegration.cpp)
target_link_libraries(test_preintegration GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_ceres_spline_helper src/test_ceres_spline_helper.cpp)
target_link_libraries(test_ceres_spline_helper GTest::gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

# benchmarks (currently doesnt work on macOS)
if(NOT APPLE)
    find_package(benchmark CONFIG REQUIRED)

    add_executable(benchmark_camera src/benchmark_camera.cpp)
    target_link_libraries(benchmark_camera benchmark::benchmark basalt::basalt-headers)
endif()

include(GoogleTest)  # for gtest_discover_test

enable_testing()

gtest_discover_tests(test_image)
gtest_discover_tests(test_spline_so3)
gtest_discover_tests(test_spline_rd)
gtest_discover_tests(test_bezier_rd)
gtest_discover_tests(test_spline_se3)
gtest_discover_tests(test_camera)
gtest_discover_tests(test_sophus)
gtest_discover_tests(test_preintegration)
gtest_discover_tests(test_ceres_spline_helper)
