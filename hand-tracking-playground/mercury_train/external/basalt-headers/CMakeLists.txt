cmake_minimum_required(VERSION 3.20)

include(CMakePackageConfigHelpers)

# Set policies for better compatibility
if(POLICY CMP0167)
  # The FindBoost module is removed in CMake 4.0
  cmake_policy(SET CMP0167 NEW)
endif()

project(basalt-headers
  VERSION 0.1.0
  DESCRIPTION "Header-only library for Basalt V-SLAM components"
  HOMEPAGE_URL "https://gitlab.com/VladyslavUsenko/basalt-headers"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CTest)

option(BASALT_HEADERS_ENABLE_INSTALL "Enable install and package export support" ON)
option(BASALT_HEADERS_BUILD_TESTING "Build basalt-headers tests when used as subproject" OFF)
# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Only apply settings when this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Enable ccache if available
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE FILEPATH "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE FILEPATH "CXX compiler launcher")
  endif()

  # Platform-specific architecture flags
  if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    if(DEFINED CXX_MARCH AND NOT CXX_MARCH STREQUAL "")
      set(BASALT_HEADERS_CXX_MARCH "${CXX_MARCH}")
    elseif(DEFINED ENV{CXX_MARCH} AND NOT "$ENV{CXX_MARCH}" STREQUAL "")
      set(BASALT_HEADERS_CXX_MARCH "$ENV{CXX_MARCH}")
    else()
      set(BASALT_HEADERS_CXX_MARCH "native")
    endif()
    set(BASALT_HEADERS_MARCH_FLAG "-march=${BASALT_HEADERS_CXX_MARCH}" CACHE STRING "Architecture flag")
    # Set build type specific flags
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN" CACHE STRING "Debug flags")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Release flags")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN" CACHE STRING "RelWithDebInfo flags")
    set(CMAKE_CXX_FLAGS_SANITIZERDEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined" CACHE STRING "Sanitizer debug flags")
    set(CMAKE_CXX_FLAGS_SANITIZERRELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined" CACHE STRING "Sanitizer relwithdebinfo flags")
  else()
    message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
    set(BASALT_HEADERS_MARCH_FLAG "")
  endif()
endif()

# Function to report dependency locations
function(report_dependency_location DEP_NAME DEP_TARGET)
  if(${DEP_NAME}_DIR)
    set(ACTUAL_DEP_CMAKE_DIR ${${DEP_NAME}_DIR})
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19.0")
    get_target_property(ACTUAL_DEP_CMAKE_DIR ${DEP_TARGET} SOURCE_DIR)
  endif()
  get_target_property(ACTUAL_DEP_INCLUDE_DIR ${DEP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Using ${DEP_NAME} from: ${ACTUAL_DEP_CMAKE_DIR} (include: ${ACTUAL_DEP_INCLUDE_DIR})")
endfunction()

# Eigen dependency
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Finding external Eigen")
  find_package(Eigen3 5.0 REQUIRED)
else()
  message(STATUS "Eigen already available")
endif()
report_dependency_location(Eigen3 Eigen3::Eigen)

# Sophus dependency
if(NOT TARGET Sophus::Sophus)
  message(STATUS "Finding external Sophus")
  find_package(Sophus REQUIRED)
else()
  message(STATUS "Sophus already available")
endif()
report_dependency_location(Sophus Sophus::Sophus)

# Cereal dependency
if(NOT TARGET cereal::cereal)
  message(STATUS "Finding external Cereal")
  find_package(cereal REQUIRED)
else()
  message(STATUS "Cereal already available")
endif()
report_dependency_location(cereal cereal::cereal)

# Main library target
add_library(basalt-headers INTERFACE)
add_library(basalt::basalt-headers ALIAS basalt-headers)

# Collect all headers automatically
file(GLOB_RECURSE BASALT_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/*.hpp
)
list(SORT BASALT_HEADER_FILES)

# Verify we found headers
list(LENGTH BASALT_HEADER_FILES BASALT_HEADER_COUNT)
message(STATUS "Found ${BASALT_HEADER_COUNT} header files")

# Set include directories
target_include_directories(basalt-headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(basalt-headers
  INTERFACE
    Eigen3::Eigen
    Sophus::Sophus
    cereal::cereal
)

# Set C++ standard requirement
target_compile_features(basalt-headers INTERFACE cxx_std_17)

# Set compile options using generator expressions
# Note: Using simpler approach that works with all CMake versions
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  target_compile_options(basalt-headers
    INTERFACE
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-exceptions>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-error=deprecated-declarations>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-unused-command-line-argument>>
      $<BUILD_INTERFACE:$<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,10>>:-Wno-misleading-indentation>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:GNU>:-Wno-error=maybe-uninitialized>>
      $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:GNU>:-Wno-error=array-bounds>>
  )
endif()

# Set -Werror separately to avoid issues with generator expressions and ccache
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # For Clang, add -Werror but allow command line argument warnings (for ccache)
    target_compile_options(basalt-headers INTERFACE $<BUILD_INTERFACE:-Werror>)
  else()
    # For GCC, just add -Werror
    target_compile_options(basalt-headers INTERFACE $<BUILD_INTERFACE:-Werror>)
  endif()
endif()


# Set architecture-specific flags more simply
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  if(BASALT_HEADERS_MARCH_FLAG)
    target_compile_options(basalt-headers INTERFACE $<BUILD_INTERFACE:${BASALT_HEADERS_MARCH_FLAG}>)
  endif()
endif()


# Build tests if requested
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BASALT_HEADERS_BUILD_TESTING) AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

# CPack configuration for package generation
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only library for Basalt V-SLAM components")
set(CPACK_PACKAGE_HOMEPAGE_URL ${PROJECT_HOMEPAGE_URL})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VENDOR "Vladyslav Usenko")

# DEB/RPM specific settings
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "eigen3-devel")

# Source package settings
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
  "/.git/"
  "/build/"
  "/.cmake_build/"
  "/cmake-build-*/"
  "/.cache/"
  "/.idea/"
  ".swp"
  ".DS_Store"
)

include(CPack)

if(BASALT_HEADERS_ENABLE_INSTALL)
  # Generate package version file
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/basalt-headers-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Create config file
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/basalt-headers-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/basalt-headers-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basalt-headers
  )

  # Install targets
  install(TARGETS basalt-headers
    EXPORT basalt-headers-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install headers
  install(DIRECTORY include/basalt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install config files
  install(EXPORT basalt-headers-targets
    FILE basalt-headers-targets.cmake
    NAMESPACE basalt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basalt-headers
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/basalt-headers-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/basalt-headers-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basalt-headers
  )
endif()
