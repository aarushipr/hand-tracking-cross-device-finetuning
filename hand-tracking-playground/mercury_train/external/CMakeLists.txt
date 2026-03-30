add_library(external_shls INTERFACE)
target_include_directories(
	external_shls INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
	)
# target_link_libraries(external_shls INTERFACE xrt-interfaces)

add_subdirectory(pybind11)
