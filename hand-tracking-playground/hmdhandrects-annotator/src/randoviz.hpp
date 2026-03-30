// Copyright Nova King, Moses Turner, probably Nick Klingensmith

#pragma once
#include "xrt/xrt_defines.h"
#include "math/m_api.h"
#include "math/m_vec3.h"

#include "stereokit.h"
#include "stereokit_ui.h"
#include <string>
using namespace sk;



inline sk::vec3
hand_to_head(handed_ handed);

inline bool
palm_facing_head(handed_ handed);

// Thickness is as a factor of size.
void
draw_axis(const pose_t &pose, float size = 0.1f, float thickness = 0.1f);

void
draw_axis(const sk::vec3 &position, float size = 0.1f, float thickness = 0.01f);

bool
draw_hand_axes();
void
draw_hand_lines();

void
text_from_vec3(sk::vec3 at, const char *hi);

void
hand_window(sk::handed_ hand, const char *hi);

void
ruler_window();