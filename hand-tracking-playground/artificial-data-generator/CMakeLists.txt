# Copyright 2022, Collabora, Ltd. SPDX-License-Identifier: BSL-1.0 Authors:
# Moses Turner <moses@collabora.com>

cmake_minimum_required(VERSION 3.10.2)
project(MonadoSubmodulerExample VERSION 0.1.0)

# set(protobuf_MODULE_COMPATIBLE TRUE)
# find_package(Protobuf CONFIG REQUIRED)
# message(STATUS "Using protobuf ${protobuf_VERSION}")

# find_package(gRPC CONFIG REQUIRED)
# message(STATUS "Using gRPC ${gRPC_VERSION}")

# We make it Debug here to get good backtraces and asserts. Then, after this, we
# set O3 so it's still fast.
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -pedantic -Wall -Wextra -Wno-unused-parameter -Werror=incompatible-pointer-types -g -march=native -O3 -fno-omit-frame-pointer"
)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -g -march=native -O3 -fno-omit-frame-pointer"
)


# I had to turn these off to get PyBind to behave. 
# Unfortunately this makes linker errors hard to debug. =/

# set(CMAKE_SHARED_LINKER_FLAGS
#     "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # -fuse-ld=mold")
# set(CMAKE_MODULE_LINKER_FLAGS
#     "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined") # -fuse-ld=mold")

set(CMAKE_CXX_STANDARD 17) # 20 breaks LDLT stuff but only for Clang aaaa
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to PIC code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create copmile_commands.json so that Intellisense works
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Get Monado's CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/monado/cmake")
list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/monado/cmake/sanitizers")

find_package(ONNXRuntime MODULE REQUIRED)

find_package(Eigen3 3 REQUIRED)
find_package(
  OpenCV REQUIRED
  COMPONENTS core
             calib3d
             highgui
             imgproc
             imgcodecs
             features2d
             video
             CONFIG)

# Make sure we have pretty colors

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-fcolor-diagnostics)
endif()

set(MONADO_SRC_XRT ${CMAKE_CURRENT_SOURCE_DIR}/monado/src/xrt)

# Set configuration options for Monado
set(XRT_FEATURE_COMPOSITOR_MAIN OFF) # we are not using this
set(XRT_FEATURE_COMPOSITOR_NULL OFF) # we are not using this
set(XRT_FEATURE_OPENXR OFF) # we aren't using this
set(XRT_FEATURE_IPC OFF)
set(XRT_FEATURE_OPENXR OFF)

set(XRT_FEATURE_RENDERDOC OFF)
set(XRT_FEATURE_SERVICE OFF)
set(XRT_FEATURE_SERVICE_SYSTEMD OFF)
set(XRT_FEATURE_STEAMVR_PLUGIN OFF) # for now. we'll see what transpires
# set(XRT_FEATURE_TRACING ON) # ditto
set(XRT_FEATURE_SLAM OFF) # Not using it
set(XRT_HAVE_BASALT_SLAM OFF)

# set(XRT_HAVE_OPENGL OFF)
set(XRT_HAVE_OPENGLES OFF)
# set(XRT_HAVE_EGL OFF) set(XRT_HAVE_VULKAN OFF)

set(XRT_FEATURE_CLIENT_DEBUG_GUI OFF)

set(XRT_HAVE_SDL2 OFF) # Don't need this - for debug UI

# No drivers!
set(XRT_BUILD_DRIVER_ANDROID OFF)
set(XRT_BUILD_DRIVER_ARDUINO OFF)
set(XRT_BUILD_DRIVER_DAYDREAM OFF)
set(XRT_BUILD_DRIVER_DEPTHAI OFF)
set(XRT_BUILD_DRIVER_EUROC OFF)
set(XRT_BUILD_DRIVER_HANDTRACKING OFF)
set(XRT_BUILD_DRIVER_TWRAP OFF)
set(XRT_BUILD_DRIVER_HDK OFF)
set(XRT_BUILD_DRIVER_HYDRA OFF)
set(XRT_BUILD_DRIVER_ILLIXR OFF)
set(XRT_BUILD_DRIVER_NS OFF)
set(XRT_BUILD_DRIVER_OHMD OFF)
set(XRT_BUILD_DRIVER_OPENGLOVES OFF)
set(XRT_BUILD_DRIVER_PSMV OFF)
set(XRT_BUILD_DRIVER_PSVR OFF)
set(XRT_BUILD_DRIVER_QWERTY OFF)
set(XRT_BUILD_DRIVER_REALSENSE OFF)
set(XRT_BUILD_DRIVER_REMOTE OFF)
set(XRT_BUILD_DRIVER_RIFT_S OFF)
set(XRT_BUILD_DRIVER_SURVIVE OFF)
set(XRT_BUILD_DRIVER_ULV2 OFF)
set(XRT_BUILD_DRIVER_VF OFF)
set(XRT_BUILD_DRIVER_VIVE OFF)
set(XRT_BUILD_DRIVER_WMR OFF)
set(XRT_BUILD_DRIVER_SIMULAVR OFF)

# There is no clean way to disable the prober/builders (as far as I can see?) So
# we enable one driver so that the legacy builder does not die
set(XRT_BUILD_DRIVER_SIMULATED ON)

set(XRT_BUILD_SAMPLES OFF)

set(BUILD_TESTING OFF)
set(BUILD_DOC OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Recurse into the main Monado distribution
add_subdirectory(monado)
add_subdirectory(src)
add_subdirectory(pybind11)



pybind11_add_module(ad4_stereographic_projection src/dataloader/dataloader_pybind.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ad4_stereographic_projection
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(ad4_stereographic_projection PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(ad4_stereographic_projection PUBLIC ${OpenCV_LIBRARIES}
          aux_math
          distort_image
          get_finger_curls
          t_ht_mercury_kine_lm
          u_uniform_distribution
          aux_util
          make_proportions)
          
add_sanitizers(ad4_stereographic_projection)


# set(_GRPC_GRPCPP gRPC::grpc++)
# set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# # Protobuf-compiler
# set(_PROTOBUF_PROTOC protoc)

# # REPEATED BOILERPLATE

# # Proto file
# get_filename_component(hw_proto "proto/artificialdata.proto" ABSOLUTE)
# get_filename_component(hw_proto_path "${hw_proto}" PATH)

# message(${hw_proto})
# message(${hw_proto_path})
# message(${_PROTOBUF_PROTOC})

# # Generated sources
# set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/artificialdata.pb.cc")
# set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/artificialdata.pb.h")
# set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/artificialdata.grpc.pb.cc")
# set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/artificialdata.grpc.pb.h")
# add_custom_command(
#   OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}"
#          "${hw_grpc_hdrs}"
#   COMMAND
#     ${_PROTOBUF_PROTOC} ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" --cpp_out
#     "${CMAKE_CURRENT_BINARY_DIR}" -I "${hw_proto_path}"
#     --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}" "${hw_proto}"
#   DEPENDS "${hw_proto}")

# # REPEATED AGAIN FOR ARTIFICIALDATA_LOADER

# # Proto file
# get_filename_component(adloader_proto "proto/artificialdata_loader.proto"
#                        ABSOLUTE)
# get_filename_component(adloader_proto_path "${adloader_proto}" PATH)

# message(${adloader_proto})
# message(${adloader_proto_path})
# message(${_PROTOBUF_PROTOC})

# # Generated sources
# set(adloader_proto_srcs
#     "${CMAKE_CURRENT_BINARY_DIR}/artificialdata_loader.pb.cc")
# set(adloader_proto_hdrs
#     "${CMAKE_CURRENT_BINARY_DIR}/artificialdata_loader.pb.h")
# set(adloader_grpc_srcs
#     "${CMAKE_CURRENT_BINARY_DIR}/artificialdata_loader.grpc.pb.cc")
# set(adloader_grpc_hdrs
#     "${CMAKE_CURRENT_BINARY_DIR}/artificialdata_loader.grpc.pb.h")
# add_custom_command(
#   OUTPUT "${adloader_proto_srcs}" "${adloader_proto_hdrs}"
#          "${adloader_grpc_srcs}" "${adloader_grpc_hdrs}"
#   COMMAND
#     ${_PROTOBUF_PROTOC} ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" --cpp_out
#     "${CMAKE_CURRENT_BINARY_DIR}" -I "${adloader_proto_path}"
#     --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#     "${adloader_proto}"
#   DEPENDS "${adloader_proto}")

# # DONE

# # Include generated *.pb.h files
# include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# message(${hw_proto_srcs})
# message(${hw_proto_hdrs})
# message(gRPC::grpc_cpp_plugin)

# # So, grpc has us use this bizarre $<TARGET_FILE thing. I don't get it. But this
# # target has to be in this file. I also think that somehow add_custom_command
# # knows ("knows?") when to run the command - doesn't run every build
# # https://github.com/IvanSafonov/grpc-cmake-example/blob/master/cmake/FindGRPC.cmake
# # is probably doing it right but I don't have time for this BS
# add_executable(server_test src/server_test.cpp ${hw_proto_srcs} ${hw_grpc_srcs})

# target_link_libraries(
#   server_test
#   PRIVATE ${OpenCV_LIBRARIES}
#           aux_math
#           StereoKitC
#           u_uniform_distribution
#           aux_util
#           pose_diversity_hand_maker
#           ${_REFLECTION}
#           ${_GRPC_GRPCPP}
#           ${_PROTOBUF_LIBPROTOBUF})

# # current_binary_dir is for grpc proto
# target_include_directories(server_test PRIVATE ${EIGEN3_INCLUDE_DIR}
#                                                ${CMAKE_CURRENT_BINARY_DIR})
# add_sanitizers(server_test)

# # Ditto - this should not be here
# add_executable(
#   dataloader_server_test src/dataloader/dataloader_server_test.cpp
#                          ${adloader_proto_srcs} ${adloader_grpc_srcs})

# target_link_libraries(
#   dataloader_server_test
#   PRIVATE ${OpenCV_LIBRARIES}
#           aux_math
#           distort_image
#           StereoKitC
#           t_ht_mercury_kine_lm
#           u_uniform_distribution
#           aux_util
#           make_proportions
#           get_finger_curls
#           ${_REFLECTION}
#           ${_GRPC_GRPCPP}
#           ${_PROTOBUF_LIBPROTOBUF})

# # current_binary_dir is for grpc proto
# target_include_directories(
#   dataloader_server_test PRIVATE ${EIGEN3_INCLUDE_DIR}
#                                  ${CMAKE_CURRENT_BINARY_DIR})
# add_sanitizers(dataloader_server_test)
