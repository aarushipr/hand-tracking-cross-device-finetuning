#include "some_defs.hpp"

void
draw_bound()
{
	sk::line_point_t pts[5];
	for (int i = 0; i < 4; i++) {
		pts[i].thickness = 0.002f;
		pts[i].color = {255, 255, 255, 255};
	}
	pts[0].pt = {-1.0f, .5f, -1.0f};
	pts[1].pt = {1.0f, .5f, -1.0f};
	pts[2].pt = {1.0f, -.5f, -1.0f};
	pts[3].pt = {-1.0f, -.5f, -1.0f};
	pts[4] = pts[0];

	sk::line_add_listv(&pts[0], 2);
	sk::line_add_listv(&pts[1], 2);
	sk::line_add_listv(&pts[2], 2);
	sk::line_add_listv(&pts[3], 2);
}

void
draw_cross(sk::vec3 pos)
{
	float size = 0.001;
	line_add(pos + sk::vec3{100000, 0, 0}, pos + sk::vec3{-100000, 0, 0}, {255, 0, 0, 255}, {255, 0, 0, 255}, size);
	line_add(pos + sk::vec3{0, 100000, 0}, pos + sk::vec3{0, -100000, 0}, {255, 0, 0, 255}, {255, 0, 0, 255}, size);
}

void
draw_rectangle(float top, float bottom, float left, float right, float height, float width, sk::color32 color)
{
	sk::vec3 one = {left, top, height};
	sk::vec3 two = {right, top, height};
	sk::vec3 three = {right, bottom, height};
	sk::vec3 four = {left, bottom, height};

	line_add(one, two, color, color, width);
	line_add(two, three, color, color, width);
	line_add(three, four, color, color, width);
	line_add(four, one, color, color, width);
}

void
draw_rectangle_cs(sk::vec2 center, sk::vec2 size, float height, float width, sk::color32 color)
{
	float left = center.x - (size.x / 2);
	float right = center.x + (size.x / 2);

	float top = center.y - (size.y / 2);
	float bottom = center.y + (size.y / 2);

	sk::vec3 one = {left, top, height};
	sk::vec3 two = {right, top, height};
	sk::vec3 three = {right, bottom, height};
	sk::vec3 four = {left, bottom, height};

	line_add(one, two, color, color, width);
	line_add(two, three, color, color, width);
	line_add(three, four, color, color, width);
	line_add(four, one, color, color, width);
}

void
text(const char *text_utf8, sk::vec3 pos, float scale = 1, bool rh = false, text_style_t tex = 0)
{
	sk::quat rot;
	if (rh) {
		rot = sk::quat_from_angles(180, 180, 0);
	} else {
		rot = sk::quat_from_angles(0, 180, 0);
	}

	sk::text_add_at(text_utf8, matrix_trs(pos, rot, {scale, scale, scale}), tex, text_align_bottom_left, text_align_bottom_left);
}

void draw_hand_box(hand_bbox_t& hand) {
			draw_rectangle_cs({hand.cx, hand.cy}, {hand.w, hand.h}, 1, .002, {255, 0, 3, 190});
		int string_acc = hand.type + 1;
		if (!((string_acc < 0) || (string_acc > 3))) {
			text(hand_class_string[hand.type + 1], {hand.cx - hand.w / 2, hand.cy - hand.h / 2, .2}, 1280, true, styles[hand.type + 1]);
		} else {
			printf("bad\n");
		}
}

void
draw_hands(std::vector<hand_bbox_t> &hands)
{
	for (hand_bbox_t hand : hands) {
		draw_hand_box(hand);

	}
}

void
viz_frames(state_t &st)
{
	float size = st.num_frames;
	for (int frame_idx = 0; frame_idx < st.num_frames; frame_idx++) {
		float fi = frame_idx;
		sk::vec3 pos0 = {-1.0f + (fi / size) * 2, .5, -.4};
		sk::vec3 pos1 = {-1.0f + ((fi + 1) / size) * 2, .5, -.4};
		sk::color32 color;
		if (st.frames[frame_idx].positions_confirmed) {
			color = {0, 255, 0, 255};
		} else {
			color = {255, 0, 0, 255};
		}
		sk::line_add(pos0, pos1, color, color, .005);
		if (frame_idx == st.curr_frame_idx) {
			color = {255, 255, 255, 255};
			sk::vec3 m = {0, -.01, 0};
			sk::line_add(pos0 + m, pos1 + m, color, color, .005);
		}
	}
}
