#include "some_defs.hpp"

void
draw_bound();

void
draw_cross(sk::vec3 pos);

void
draw_rectangle(float top, float bottom, float left, float right, float height, float width, sk::color32 color);

void
draw_rectangle_cs(sk::vec2 center, sk::vec2 size, float height, float width, sk::color32 color);

void
text(const char *text_utf8, sk::vec3 pos, float scale = 1, bool rh = false, text_style_t tex = 0);

void
draw_hand_box(hand_bbox_t &hand);

void
draw_hands(std::vector<hand_bbox_t> &hands);

void
viz_frames(state_t &st);
