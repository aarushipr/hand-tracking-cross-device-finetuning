vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

set(SOURCE_PATH "${CURRENT_PORT_DIR}/../..")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBASALT_HEADERS_ENABLE_INSTALL=ON
        -DBASALT_HEADERS_BUILD_TESTING=OFF
        -DBUILD_TESTING=OFF
        -DBASALT_BUILTIN_EIGEN=OFF
        -DBASALT_BUILTIN_SOPHUS=OFF
        -DBASALT_BUILTIN_CEREAL=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/basalt-headers)
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL
    "${CMAKE_CURRENT_LIST_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
file(INSTALL
    "${SOURCE_PATH}/LICENSE"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright
)
