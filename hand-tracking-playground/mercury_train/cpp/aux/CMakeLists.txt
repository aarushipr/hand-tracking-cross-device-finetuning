add_library(u_random_distribution u_random_distribution.cpp u_random_distribution.h)
target_link_libraries(u_random_distribution aux-includes aux_util aux_tracking)

# note: pose_csv and randoviz are header-only, and get included through
# htdev_includes.
