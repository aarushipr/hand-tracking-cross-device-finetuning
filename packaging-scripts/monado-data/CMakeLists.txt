# Copyright 2023, Collabora, Ltd.
# SPDX-License-Identifier: MIT or BSL-1.0 or Apache-2.0

cmake_minimum_required(VERSION 3.10.2)

# Used in a lot of places.
set(VER_MAJOR "21")
set(VER_MINOR "0")
set(VER_PATCH "0")
set(VER "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")

project(monado-data VERSION "${VER}")

include(GNUInstallDirs)

set(CPACK_PACKAGE_NAME "monado-data")
set(CPACK_PACKAGE_VENDOR "Sentients of the Monado project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ML inference models for the Monado Project.")
set(CPACK_PACKAGE_VERSION "${VER}")
set(CPACK_PACKAGE_VERSION_MAJOR "${VER_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VER_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VER_PATCH}")
set(CPACK_PACKAGE_CONTACT "Jakob Bornecrantz <jakob@collabora.com>")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_NAME "monado-data")

set(DETECTION_MODEL "hand/grayscale_detection_160x160.onnx")
set(KEYPOINT_MODEL "hand/grayscale_keypoint_jan18.onnx")

install(FILES "${KEYPOINT_MODEL}" DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/monado/hand-tracking-models")
install(FILES "${DETECTION_MODEL}" DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/monado/hand-tracking-models")

include(CPack)
