add_subdirectory(kine_lm)

add_library(make_proportions STATIC make_hand_proportions.cpp make_hand_proportions.hpp)

target_link_libraries(
	make_proportions
	PRIVATE
		${OpenCV_LIBRARIES}
		htdev_includes
		aux_math
		StereoKitC # uncomment if u want
		# stereokit
		data_generator_kine_lm
		aux_util
	)

add_executable(pose_diversity_visualizer pose_diversity_test.cpp)

target_link_libraries(
	pose_diversity_visualizer
	PRIVATE
		${OpenCV_LIBRARIES}
		htdev_includes
		aux_math
		StereoKitC # uncomment if u want
		# stereokit
		data_generator_kine_lm
		aux_util
		pose_diversity_hand_maker
	)

target_include_directories(pose_diversity_visualizer PRIVATE ${EIGEN3_INCLUDE_DIR})
add_sanitizers(pose_diversity_visualizer)

add_library(pose_diversity_hand_maker pose_diversity_hand_maker.cpp pose_diversity_hand_maker.h)
target_link_libraries(
	pose_diversity_hand_maker
	htdev_includes
	aux-includes
	aux_util
	u_random_distribution
	aux_tracking
	data_generator_kine_lm
	aux_util
	make_proportions
	)

add_executable(data_generator data_generator.cpp)

target_link_libraries(
	data_generator
	PRIVATE
		${OpenCV_LIBRARIES}
		aux_math
		StereoKitC
		u_random_distribution
		aux_util
		pose_diversity_hand_maker
		external_shls
	)

target_include_directories(data_generator PRIVATE ${EIGEN3_INCLUDE_DIR})
add_sanitizers(data_generator)
