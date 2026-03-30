# Copyright Collabora, Ltd., 2021.

# SPDX-License-Identifier: BSL-1.0

# Author: Moses Turner


add_library(
    do_hand STATIC 
    do_hand.cpp
)
# set_property(TARGET do_hand PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_link_libraries(do_hand PUBLIC aux-includes  Ceres::ceres aux_os aux_util)

target_include_directories(do_hand SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})



add_executable(raygun ray.cpp)
target_link_libraries(raygun PRIVATE StereoKitC::StereoKitC do_hand aux_os aux_util)

# set_property(TARGET raygun PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(raygun PRIVATE ${EIGEN3_INCLUDE_DIR})

add_sanitizers(raygun)

install(TARGETS raygun)



add_executable(allatonce test_all_at_once.cpp)
target_link_libraries(allatonce PRIVATE StereoKitC::StereoKitC do_hand aux_os aux_util)

# # set_property(TARGET allatonce PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(allatonce PRIVATE ${EIGEN3_INCLUDE_DIR})

add_sanitizers(allatonce)

install(TARGETS allatonce)



# add_executable(threequel kinematic_troogaloo.cpp)
# target_link_libraries(threequel PRIVATE StereoKitC Ceres::ceres aux_os aux_util)

# set_property(TARGET threequel PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# target_include_directories(threequel PRIVATE ${EIGEN3_INCLUDE_DIR})

# add_sanitizers(threequel)

# install(TARGETS threequel)

