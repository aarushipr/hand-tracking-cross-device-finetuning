# Copyright Collabora, Ltd., 2021.

# SPDX-License-Identifier: BSL-1.0

# Author: Moses Turner

add_executable(human_annotator human_annotator_main.cpp some_defs.hpp 2d_viz.hpp 2d_viz.cpp randoviz.hpp randoviz.cpp)
add_sanitizers(human_annotator)

target_link_libraries(
  human_annotator
  PRIVATE
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  ${OpenCV_LIBRARIES}
  aux_math
  aux_util
  aux_tracking)

target_include_directories(human_annotator PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})

install(TARGETS human_annotator)


add_executable(redactor redactor.cpp some_defs.hpp 2d_viz.hpp 2d_viz.cpp randoviz.hpp randoviz.cpp)
add_sanitizers(redactor)

target_link_libraries(
  redactor
  PRIVATE
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  ${OpenCV_LIBRARIES}
  aux_math
  aux_util
  aux_tracking)

target_include_directories(redactor PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})

install(TARGETS redactor)



add_executable(test clitest.cpp)
add_sanitizers(test)

target_link_libraries(
  test
  PRIVATE
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  aux_math
  aux_util
  aux_tracking)

target_include_directories(test PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})

install(TARGETS test)



add_executable(machine_annotator machine_annotator/machine_annotator_main.cpp)
add_sanitizers(machine_annotator)

target_link_libraries(
  machine_annotator
  PRIVATE
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  aux_math
  aux_util
  ONNXRuntime::ONNXRuntime
  aux_tracking)

target_include_directories(machine_annotator PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})

install(TARGETS machine_annotator)
