add_subdirectory(aux)
add_subdirectory(data_generator)
add_subdirectory(dataloader)
add_subdirectory(pose_recorders)
add_subdirectory(dataset_playback)
add_subdirectory(dataset_annotator)

add_library(htdev_includes INTERFACE)
target_include_directories(
	htdev_includes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
	)
target_link_libraries(htdev_includes INTERFACE external_shls)
