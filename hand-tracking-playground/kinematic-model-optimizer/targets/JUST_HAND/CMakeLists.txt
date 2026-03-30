# Copyright Collabora, Ltd., 2021.

# SPDX-License-Identifier: BSL-1.0

# Author: Moses Turner



add_executable(threequel kinematic_troogaloo.cpp)
target_link_libraries(threequel PRIVATE StereoKitC::StereoKitC Ceres::ceres aux_os aux_util)

set_property(TARGET threequel PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(threequel PRIVATE ${EIGEN3_INCLUDE_DIR})

add_sanitizers(threequel)

install(TARGETS threequel)

