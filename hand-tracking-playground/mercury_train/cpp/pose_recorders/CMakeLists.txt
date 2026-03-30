add_executable(wristpose_recorder wrist_elbow_pose.cpp)

target_link_libraries(
	wristpose_recorder
	PRIVATE
		-L/usr/local/lib
		StereoKitC
		${OpenCV_LIBRARIES}
		aux-includes
		aux_util
		htdev_includes
	)

target_include_directories(wristpose_recorder PRIVATE ${OpenCV_INCLUDE_DIRS})
