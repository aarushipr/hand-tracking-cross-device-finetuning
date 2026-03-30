// Copyright Nova King, Moses Turner, probably Nick Klingensmith


#include "xrt/xrt_defines.h"
#include "math/m_api.h"
#include "math/m_vec3.h"

#include "stereokit.h"
#include "stereokit_ui.h"
#include <string>
using namespace sk;



inline sk::vec3
hand_to_head(handed_ handed)
{
	return input_head()->position - input_hand(handed)->palm.position;
}

inline bool
palm_facing_head(handed_ handed)
{
	const sk::hand_t *hand = input_hand(handed);
	if (!(hand->tracked_state & button_state_active)) {
		return false;
	}

	xrt_vec3 palm_direction;
	math_quat_rotate_vec3((xrt_quat *)&hand->palm.orientation, (xrt_vec3 *)&vec3_forward, (xrt_vec3 *)&palm_direction);
	palm_direction = m_vec3_normalize(palm_direction);
	sk::vec3 hand_to_head_direction = sk::vec3_normalize(hand_to_head(handed));

	return m_vec3_dot(palm_direction, *(xrt_vec3 *)&hand_to_head_direction) > 0.5f;
}

void
_draw_axis(const matrix &mat, float size, float thickness)
{
	sk::vec3 pos = matrix_extract_translation(mat);
	line_add(pos, matrix_transform_dir(mat, {1, 0, 0}) * size + pos, {255, 0, 0, 255}, {255, 0, 0, 255}, thickness);
	line_add(pos, matrix_transform_dir(mat, {0, 1, 0}) * size + pos, {0, 255, 0, 255}, {0, 255, 0, 255}, thickness);
	line_add(pos, matrix_transform_dir(mat, {0, 0, 1}) * size + pos, {0, 0, 255, 255}, {0, 0, 255, 255}, thickness);
}


// Thickness is as a factor of size.
void
draw_axis(const pose_t &pose, float size = 0.1f, float thickness = 0.1f)
{
	// matrix mat = ;
	_draw_axis(pose_matrix(pose), size, thickness * size);

	// line_add(pose.position, matrix_mul_direction(pose_matrix_, {1.0f, 0, 0}) * size + pose.position,
	//          {255, 0, 0, 255}, {255, 0, 0, 255}, thickness);
	// line_add(pose.position, matrix_mul_direction(pose_matrix_, {0, 1.0f, 0}) * size + pose.position,
	//          {0, 255, 0, 255}, {0, 255, 0, 255}, thickness);
	// line_add(pose.position, matrix_mul_direction(pose_matrix_, {0, 0, 1.0f}) * size + pose.position,
	//          {0, 0, 255, 255}, {0, 0, 255, 255}, thickness);
}

void
draw_axis(const sk::vec3 &position, float size = 0.1f, float thickness = 0.01f)
{
	_draw_axis(matrix_trs(position), size, thickness);
}

bool
draw_hand_axes()
{
	for (int i = 0; i < handed_max; i++) {
		const hand_t *hand = input_hand((handed_)i);
		if (hand->tracked_state == button_state_inactive) {
			continue;
		}

		for (int finger = 0; finger < 5; finger++) {
			for (int joint = 0; joint < 5; joint++) {
				hand_joint_t joint_ = hand->fingers[finger][joint];
				pose_t joint_pose = {joint_.position, joint_.orientation};

				draw_axis(joint_pose, 0.015f, 0.002);
			}
		}

		pose_t palm_pose = hand->palm;
		draw_axis(palm_pose, 0.015f, 0.002);

		pose_t wrist_pose = hand->wrist;
		// log_diagf("I think wrist is at %f %f %f", wrist_pose.position.x,
		// wrist_pose.position.y, wrist_pose.position.z);
		draw_axis(wrist_pose, 15.0f, 0.002);
		vec3 wrist = wrist_pose.position;
		vec3 middle_distal = hand->fingers[2][1].position;
		vec3 sub = wrist - middle_distal;
	}
	return true;
}

void
draw_hand_lines()
{
	for (int side = 0; side < handed_max; side++) {
		const hand_t *hand = input_hand((handed_)side);
		if (hand->tracked_state == button_state_inactive)
			continue;

		for (int finger = 0; finger < 5; finger++) {
			for (int joint = 0; joint < 4; joint++) {
				hand_joint_t joint0 = hand->fingers[finger][joint];
				hand_joint_t joint1 = hand->fingers[finger][joint + 1];
				float hue0 = ((finger * 5) + joint) / 24.0f;
				float hue1 = ((finger * 5) + joint + 1) / 24.0f;
				if (finger == 0 && joint == 0) {
					line_add(hand->wrist.position, joint0.position, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)), color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.002);
				} else {
					line_add(joint0.position, joint1.position, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)), color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.002);
				}
			}
		}
	}
}

void
text_from_vec3(sk::vec3 at, const char *hi)
{
	vec3 left_position = hierarchy_to_world_point(at);

	hierarchy_set_enabled(false);



	vec3 head_position = input_head()->position;
	pose_t pose;
	pose.position = (left_position * .9) + (head_position * .1);
	pose.orientation = quat_lookat(left_position, head_position);
	text_add_at(hi, pose_matrix(pose));

  hierarchy_set_enabled(true);
}

void
text_from_vec3(sk::vec3 at, float size, const char *hi)
{
	vec3 left_position = hierarchy_to_world_point(at);

	hierarchy_set_enabled(false);



	vec3 head_position = input_head()->position;
	pose_t pose;
	pose.position = (left_position * .9) + (head_position * .1);
	pose.orientation = input_head()->orientation;// quat_lookat(left_position, head_position);
	text_add_at(hi, pose_matrix(pose, {-size, size, size}));

  hierarchy_set_enabled(true);
}

void
hand_window(sk::handed_ hand, const char *hi)
{
	const hand_t *hande = input_hand(hand);
	if (hande->tracked_state == button_state_inactive)
		return;
	vec3 left_position = hande->palm.position;

	vec3 head_position = input_head()->position;
	pose_t pose;
	pose.position = (left_position * .9) + (head_position * .1);
	pose.orientation = quat_lookat(left_position, head_position);
	// quat_mul(head_potition, quat_from_angles(0, 180, 0)); // head_potition;
	if (input_hand(hand)->tracked_state == button_state_active) {
		hierarchy_push(pose_matrix(pose));
		text_add_at(hi, matrix_identity);
		hierarchy_pop();
		// ui_window_begin(hi, pose, vec2{0, 0});
		// ui_window_end();
	}
}

void
ruler_window()
{
	static pose_t window_pose = pose_t{{0, 0, 0.5f}, quat_identity};
	ui_handle_begin("Ruler", window_pose, bounds_t{vec3_zero, vec3{30 * cm2m, 4 * cm2m, 1 * cm2m}}, true, ui_move_exact);
	color32 color = color_to_32(color_hsv(0.6f, 0.5f, 1, 1));
	text_add_at("Centimeters", matrix_trs(vec3{14.5f * cm2m, -1.5f * cm2m, -0.6f * cm2m}, quat_identity, vec3{0.3f, 0.3f, 0.3f}), 0, text_align_bottom_left);
	for (int d = 0; d <= 60; d++) {
		float x = d / 2.0f;
		float size = (d % 2 == 0) ? 1.0f : 0.15f;
		line_add(vec3{(15 - x) * cm2m, 2 * cm2m, -0.6f * cm2m}, vec3{(15 - x) * cm2m, (2 - size) * cm2m, -0.6f * cm2m}, color, color, 0.5f * mm2m);

		if (d % 2 == 0 && d / 2 != 30) {
			text_add_at(std::to_string(d / 2).c_str(), matrix_trs(vec3{(15 - x - 0.1f) * cm2m, (2 - size) * cm2m, -0.6f * cm2m}, quat_identity, vec3{0.2f, 0.2f, 0.2f}), 0,
			            text_align_bottom_left);
		}
	}
	ui_handle_end();
}