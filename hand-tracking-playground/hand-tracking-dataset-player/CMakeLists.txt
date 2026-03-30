# Copyright 2018-2020, Collabora, Ltd.
# SPDX-License-Identifier: BSL-1.0

# If building for android, must call CMake with something like:
# cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake .. -DANDROID_PLATFORM=24
#
# You need the toolchain file to cause CMake to build for Android
# You need a new enough platform level to get Vulkan headers - 24 known to work

cmake_minimum_required(VERSION 3.10.2)
project(MonadoBasedRuntimeExample VERSION 0.1.0)

if(NOT MSVC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wall -Wextra -Wno-unused-parameter -Werror=incompatible-pointer-types")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to PIC code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/monado/cmake/sanitizers")
# find_package(Sanitizers MODULE)


set(MONADO_SRC_XRT ${CMAKE_CURRENT_SOURCE_DIR}/monado/src/xrt)

### Set configuration options for Monado
set(XRT_FEATURE_COMPOSITOR_MAIN OFF) # we are replacing this
set(XRT_FEATURE_SERVICE OFF) # in-process for now, change in future
set(XRT_FEATURE_OPENXR OFF) # we need to replace this to pull together a different combination of modules
set(XRT_HAVE_SDL2 OFF) # Don't need this - for debug UI

### Modify this to change the name of .so containing the runtime
set(RUNTIME_BARE_SUFFIX custom)

# Package name needs to be known by the native code itself.
# Can be overridden from outside/command line
if(ANDROID AND NOT XRT_ANDROID_PACKAGE)
	if(NOT BASE_APP_ID)
		message("WARNING: Make sure to pass -DBASE_APP_ID from gradle to CMake, or the binary won't be able to find itself!")
		set(BASE_APP_ID "invalid_placeholder_missing_base_app_id")
	endif()
	if(XRT_FEATURE_SERVICE)
		set(XRT_ANDROID_PACKAGE "${BASE_APP_ID}.out_of_process")
	else()
		set(XRT_ANDROID_PACKAGE "${BASE_APP_ID}.in_process")
	endif()
endif()

# Make sure we have pretty colours
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)

if ("${FORCE_COLORED_OUTPUT}")
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options (-fdiagnostics-color=always)
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		add_compile_options (-fcolor-diagnostics)
	endif ()
endif ()

### Recurse into the main Monado distribution
add_subdirectory(monado)

### Your Monado-based builds go here.

# The directory structure here intentionally mirrors that of Monado
# within its src/xrt directory, for ease of organization and
# familiarity.


# Your custom drivers
# add_subdirectory(drivers)

# Your custom compositor
# add_subdirectory(compositor)

# If you don't use the standard compositor or need to inject a driver,
# you will need to build your own openxr target (a very minimal bit of glue code)
find_package(Eigen3 3 REQUIRED)
add_subdirectory(targets)


