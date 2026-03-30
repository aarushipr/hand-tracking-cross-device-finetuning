# Copyright Moses Turner, 2021. SPDX-License-Identifier: BSL-1.0
cmake_minimum_required(VERSION 3.10)

# You probably want C++17 for now
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# So that Intellisense/clangd know how your project is compiled
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")


# Uncomment all these if you want address sanitizers:

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -march=native -O3 -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -march=native -O3  -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_VERBOSE_MAKEFILE ON)

project(
  sk-webcam-example
  VERSION "0.0.1"
  LANGUAGES CXX C)

find_package(StereoKitC REQUIRED)
find_package(
  OpenCV
  COMPONENTS core
             calib3d
             highgui
             imgproc
             imgcodecs
             features2d
             video
             videoio
  REQUIRED)

add_library(our-pthreads INTERFACE)
if(WIN32)
  find_package(pthreads_windows REQUIRED)
  target_link_libraries(our-pthreads INTERFACE PThreads4W::PThreads4W_CXXEXC)
else()
  set(CMAKE_THREAD_PREFER_PTHREAD ON)
  find_package(Threads)
  target_link_libraries(our-pthreads INTERFACE Threads::Threads)
endif()


if("${FORCE_COLORED_OUTPUT}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()




add_library(blendernexus STATIC src/blendernexus.cpp src/blendernexus.hpp src/u_uniform_distribution.h src/u_uniform_distribution.cpp src/distort_image.hpp src/distort_image.cpp  )

target_link_libraries(blendernexus PRIVATE -L/usr/local/lib 
                                         ${OpenCV_LIBRARIES} aux-includes aux_util aux_tracking our-pthreads stdc++fs)

target_include_directories(blendernexus PRIVATE ${OpenCV_INCLUDE_DIRS})




add_executable(dataset_generator src/generate_many.cpp src/blendernexus.hpp src/u_uniform_distribution.h src/u_uniform_distribution.cpp)

target_link_libraries(dataset_generator PRIVATE blendernexus 
                                         ${OpenCV_LIBRARIES} aux-includes aux_util aux_tracking our-pthreads stdc++fs)

target_include_directories(dataset_generator PRIVATE ${OpenCV_INCLUDE_DIRS})


add_executable(dataset_generator_multithreaded src/generate_many_multithreaded.cpp src/blendernexus.hpp src/u_uniform_distribution.h src/u_uniform_distribution.cpp)

target_link_libraries(dataset_generator_multithreaded PRIVATE blendernexus 
                                         ${OpenCV_LIBRARIES} aux-includes aux_util aux_tracking our-pthreads stdc++fs)

target_include_directories(dataset_generator_multithreaded PRIVATE ${OpenCV_INCLUDE_DIRS})



add_subdirectory(monado)
