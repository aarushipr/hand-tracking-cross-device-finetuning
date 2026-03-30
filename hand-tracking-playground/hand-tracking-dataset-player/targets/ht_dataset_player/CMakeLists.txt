# Copyright Collabora, Ltd., 2021.

# SPDX-License-Identifier: BSL-1.0

# Author: Moses Turner

add_executable(descent-into-madness descent_main.cpp randoviz.cpp) # some_defs.hpp 2d_viz.hpp 2d_viz.cpp randoviz.hpp randoviz.cpp)
add_sanitizers(descent-into-madness)

target_link_libraries(
  descent-into-madness
  PRIVATE
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  ${GST_LIBRARIES}
  ${OpenCV_LIBRARIES}
  aux_math
  aux_util
  aux_tracking
  drv_euroc
  t_ht_mercury
  t_ht_mercury_includes
  )

target_include_directories(descent-into-madness PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})

install(TARGETS descent-into-madness)
