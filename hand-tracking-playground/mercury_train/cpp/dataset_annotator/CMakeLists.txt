
add_executable(annotator annotator_main.cpp)

target_link_libraries(annotator PRIVATE -L/usr/local/lib StereoKitC external_shls aux_util)