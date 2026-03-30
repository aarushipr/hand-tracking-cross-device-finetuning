add_library(distort_image STATIC distort_image.cpp)
target_include_directories(distort_image PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(
	distort_image
	aux-includes
	aux_util
	aux_math
	aux_tracking
	)

add_library(get_finger_curls STATIC get_finger_curls.cpp)
target_include_directories(get_finger_curls PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(
	get_finger_curls
	aux-includes
	aux_util
	aux_math
	aux_tracking
	)

# This is the actual important thing we're making.
pybind11_add_module(ad4_stereographic_projection dataloader_pybind.cpp)

target_compile_definitions(
	ad4_stereographic_projection PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO}
	)
target_include_directories(ad4_stereographic_projection PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(
	ad4_stereographic_projection
	PUBLIC
		${OpenCV_LIBRARIES}
		aux_math
		htdev_includes
		distort_image
		get_finger_curls
		data_generator_kine_lm
		u_random_distribution
		aux_util
		make_proportions
	)

add_sanitizers(ad4_stereographic_projection)
