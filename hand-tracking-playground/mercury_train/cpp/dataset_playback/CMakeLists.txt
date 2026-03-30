# Copyright Collabora, Ltd., 2023.

# SPDX-License-Identifier: BSL-1.0

# Author: Moshi Turner

# add_executable(descent-into-madness descent_main.cpp randoviz.cpp) # some_defs.hpp 2d_viz.hpp 2d_viz.cpp randoviz.hpp randoviz.cpp)
add_executable(descent-into-madness sync_runner.cpp sync_runner.hpp
                                    load_basalt_calibration.cpp load_basalt_calibration.hpp
)

target_link_libraries(
  descent-into-madness
  PRIVATE
  # st_gui
  xrt-pthreads
  -L/usr/local/lib
  StereoKitC
  ${GST_LIBRARIES}
  ${OpenCV_LIBRARIES}
  aux_math
  aux_vive
  aux_util
  aux_tracking
  drv_euroc
  drv_includes
  t_ht_mercury
  t_ht_mercury_includes
  aux_util_debug_gui
  st_gui
  target_instance_no_comp
  )

target_include_directories(descent-into-madness PRIVATE ${GST_INCLUDE_DIRS}
                                                   ${EIGEN3_INCLUDE_DIR})
add_sanitizers(descent-into-madness)

install(TARGETS descent-into-madness)
