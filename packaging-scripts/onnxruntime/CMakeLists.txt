# Copyright 2023, Collabora, Ltd.
# SPDX-License-Identifier: MIT or BSL-1.0 or Apache-2.0

cmake_minimum_required(VERSION 3.10.2)

# Used in a lot of places.
set(VER_MAJOR "1")
set(VER_MINOR "14")
set(VER_PATCH "1")
set(VER "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")

project(CPackONNX VERSION "${VER}")

include(GNUInstallDirs)

set(CPACK_PACKAGE_NAME "libonnxruntime")
set(CPACK_PACKAGE_VENDOR "Sentients of the Monado project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A very ugly package for onnxruntime")
set(CPACK_PACKAGE_VERSION "${VER}")
set(CPACK_PACKAGE_VERSION_MAJOR "${VER_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VER_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VER_PATCH}")
set(CPACK_PACKAGE_CONTACT "Jakob Bornecrantz <jakob@collabora.com>")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_NAME "libonnxruntime")

# Seems to be the only thing it depends on (Ubuntu 22.04).
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libstdc++6")

set(PKG "onnxruntime-linux-x64-${VER}")

set(C_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PKG}/include/onnxruntime_c_api.h")
set(CXX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PKG}/include/onnxruntime_cxx_api.h")
set(SYM_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PKG}/lib/libonnxruntime.so")
set(LIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PKG}/lib/libonnxruntime.so.${VER}")
set(ZIP_FILE "${CMAKE_SOURCE_DIR}/${PKG}.tgz")
set(PC_FILE "${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc")
set(GEN_FILES ${LIB_FILE} ${SYM_FILE} ${C_FILE} ${CXX_FILE})


configure_file(${CMAKE_SOURCE_DIR}/libonnxruntime.pc.cmake_in ${PC_FILE} @ONLY)

add_custom_command(
	OUTPUT ${GEN_FILES}
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${ZIP_FILE}
	DEPENDS ${ZIP_FILE}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Unpacking ${ZIP_FILE}"
	VERBATIM
	)

add_custom_target(onnxruntime_unpack ALL DEPENDS ${GEN_FILES})

install(FILES ${PC_FILE} DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(FILES ${LIB_FILE} ${SYM_FILE} DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${C_FILE} ${CXX_FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CPack)

