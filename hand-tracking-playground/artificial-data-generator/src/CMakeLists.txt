add_subdirectory(kine_lm)

add_library(make_proportions STATIC make_hand_proportions.cpp
                                    make_hand_proportions.hpp)

target_link_libraries(make_proportions)
target_link_libraries(
  make_proportions
  PRIVATE ${OpenCV_LIBRARIES} aux_math StereoKitC # uncomment if u want
                                                  # stereokit
          t_ht_mercury_kine_lm aux_util)

add_executable(test pose_diversity_test.cpp)

target_link_libraries(
  test
  PRIVATE ${OpenCV_LIBRARIES} aux_math StereoKitC # uncomment if u want
                                                  # stereokit
          t_ht_mercury_kine_lm aux_util pose_diversity_hand_maker)

target_include_directories(test PRIVATE ${EIGEN3_INCLUDE_DIR})
add_sanitizers(test)


add_library(u_uniform_distribution u_uniform_distribution.cpp u_uniform_distribution.h)
target_link_libraries(u_uniform_distribution aux-includes aux_util aux_tracking)

add_library(pose_diversity_hand_maker pose_diversity_hand_maker.cpp pose_diversity_hand_maker.h)
target_link_libraries(pose_diversity_hand_maker aux-includes aux_util aux_tracking t_ht_mercury_kine_lm aux_util make_proportions)


add_library(distort_image STATIC dataloader/distort_image.cpp)
target_include_directories(distort_image PRIVATE ${EIGEN3_INCLUDE_DIR} )
target_link_libraries(distort_image aux-includes aux_util aux_math aux_tracking )

add_library(get_finger_curls STATIC dataloader/get_finger_curls.cpp)
target_include_directories(get_finger_curls PRIVATE ${EIGEN3_INCLUDE_DIR} )
target_link_libraries(get_finger_curls aux-includes aux_util aux_math aux_tracking )



add_subdirectory(pose_recorders)