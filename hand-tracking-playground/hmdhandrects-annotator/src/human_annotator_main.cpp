#include "math/m_api.h"
#include "some_defs.hpp"
#include "2d_viz.hpp"
// #include "../monado/src/xrt/drivers/ht/templates/NaivePermutationSort.hpp"
#include "../monado/src/xrt/tracking/hand/old_rgb/templates/NaivePermutationSort.hpp"
#include "util/u_logging.h"
#include "CLI11.hpp"
#include <stereokit.h>


// const char *view_keys[] = {"left", "right"};

text_style_t styles[3];

// const char *hand_class_string[] = {"UNKNOWN", "EGO_LEFT", "EGO_RIGHT", "OTHER_LEFT", "OTHER_RIGHT", "REDACTED"};

state_t st = {};

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


void
step_frame(int amount)
{

	int old_frame = st.curr_frame_idx;

	st.curr_frame_idx += amount;

	st.curr_frame_idx = fmin(fmax(st.curr_frame_idx, 0), st.num_frames - 1);

	if (st.curr_frame_idx == old_frame) {
		return;
	}

	st.this_frame = &st.frames[st.curr_frame_idx];

	for (int view = 0; view < 2; view++) {
		st.view[view].img_filename = fs::path(st.this_frame->views[view].filename);
		st.view[view].img_mat = cv::imread(st.paths.root / st.view[view].img_filename, cv::IMREAD_GRAYSCALE);
		st.view[view].img_mat.convertTo(st.view[view].img_mat, CV_32F, 1 / 255.0);

		cv::Mat mean;
		cv::Mat stddev;
		cv::meanStdDev(st.view[view].img_mat, mean, stddev);

		st.view[view].img_mat *= 0.3 / stddev.at<double>(0, 0);

		// Calculate it again; mean has changed. Yes we odn't need to but it's easy
		cv::meanStdDev(st.view[view].img_mat, mean, stddev);
		st.view[view].img_mat += (0.5 - mean.at<double>(0, 0));


		cv::cvtColor(st.view[view].img_mat, st.view[view].img_mat, cv::COLOR_BGR2RGBA);

		tex_set_colors(st.view[view].img_tex, 1280, 800, st.view[view].img_mat.data);
	}
	// printf("%s\n", cJSON_Print(st.this_frame));
}

void
update_global_mouse()
{
	const mouse_t *mouse = input_mouse();
	ray_t r;
	ray_from_mouse(mouse->pos, r);
	st.global_mouse_position.x = r.pos.x;
	st.global_mouse_position.y = r.pos.y;
}

void
manage_pan_zoom()
{
	const sk::mouse_t *mouse = input_mouse();
	if (mouse->scroll_change != 0.0) {
		sk::matrix root = sk::render_get_cam_root();
		float s = 0.88;
		if (mouse->scroll_change < 0.0) {
			s = 1 / s; // 1.08;
		}

		sk::matrix matrix_scale = sk::matrix_s({s, s, 1});
		sk::vec3 trans = sk::vec3{st.global_mouse_position.x, st.global_mouse_position.y, 0} * (1 - s);
		trans.z = 0;
		sk::matrix matrix_translate = sk::matrix_t(trans);
		sk::matrix transform;

		sk::matrix_mul(matrix_scale, matrix_translate, transform);

		sk::matrix new_;

		sk::matrix_mul(root, transform, new_);

		sk::render_set_cam_root(new_);

		// We just moved the camera, recalculate mouse position!
		update_global_mouse();
	}

	else if (sk::input_key(sk::key_mouse_center)) {
		sk::matrix root = sk::render_get_cam_root();
		ray_t old;
		ray_from_mouse(mouse->pos - mouse->pos_change, old);
		sk::vec2 move = sk::vec2{old.pos.x, old.pos.y} - st.global_mouse_position;
		sk::matrix new_;
		sk::matrix_mul(root, sk::matrix_trs({move.x, move.y, 0}), new_);
		sk::render_set_cam_root(new_);

		// We just moved the camera, recalculate mouse position!
		update_global_mouse();
	}
}

void
manage_screen_resize()
{
	system_info_t info = sk_system_info();

	if (info.display_width == st.old_display_width && info.display_height == st.old_display_height) {
		return;
	}

	st.old_display_width = info.display_width;
	st.old_display_height = info.display_height;

	if ((float)info.display_width / (float)info.display_height > 2.0f) {
		render_set_ortho_size(1.0f);
	} else {
		float size = (2.0f / ((float)info.display_width / (float)info.display_height));
		render_set_ortho_size(size);
	}
}

bool
mouse_in_bbox(hand_bbox_t &hand, sk::vec3 mouse_position, float outer_border = 2.0f)
{
	return (fabs(hand.cx - mouse_position.x) < ((hand.w / 2) + outer_border)) && (fabs(hand.cy - mouse_position.y) < ((hand.h / 2) + outer_border));
}

void
remove_hands(view_t *hands, sk::vec3 mouse_position)
{
	printf("startr\n");
	size_t curr_idx = 0;
	while (true) {
		if (curr_idx == hands->hands.size()) {
			break;
		}
		hand_bbox_t *hand = &hands->hands[curr_idx];

		printf("remove_hands: idx %zu size %zu\n", curr_idx, hands->hands.size());

		if (mouse_in_bbox(*hand, mouse_position)) {
			hands->hands.erase(hands->hands.begin() + curr_idx);
			continue;
		}
		curr_idx++;
	}
}

void
remove_misclicks(view_t *hands)
{
	printf("startr\n");
	size_t curr_idx = 0;
	while (true) {
		if (curr_idx == hands->hands.size()) {
			break;
		}
		hand_bbox_t *hand = &hands->hands[curr_idx];

		printf("remove_misclicks: idx %zu size %zu\n", curr_idx, hands->hands.size());

		if ((fabs(hand->w) < 2.0f) || (fabs(hand->h) < 2.0f)) {
			hands->hands.erase(hands->hands.begin() + curr_idx);
			continue;
		}
		curr_idx++;
	}
}

int
find_boundary(int start, int dir)
{

	bool start_state = st.frames[start].positions_confirmed;

	int curr = start;

	int i = 0;

	for (; (curr >= 0) && (curr < st.num_frames); curr += dir) {
		if ((st.frames[curr].positions_confirmed != (start_state)) || (st.linking_two_boxes.active && curr == st.linking_two_boxes.start_frame_idx)) {
			if (i < 2) {
				start_state = !start_state;
			} else {
				break;
			}
		}
		i++;
	}
	return (curr - start);
}

void
step_frames()
{
	bool fwd = (sk::input_key(sk::key_d) & sk::button_state_just_inactive) || (sk::input_key(sk::key_right) & sk::button_state_just_inactive);
	bool bwd = (sk::input_key(sk::key_a) & sk::button_state_just_inactive) || (sk::input_key(sk::key_left) & sk::button_state_just_inactive);

	if (fwd == bwd) {
		// They pressed both keys or no keys; do nothing!
		return;
	}

	bool big_step = (sk::input_key(sk::key_ctrl));
	bool boundary_step = (sk::input_key(sk::key_shift));

	int dir = 1;
	if (bwd) {
		dir = -1;
	}

	if (boundary_step) {
		step_frame(find_boundary(st.curr_frame_idx, dir));
	} else if (big_step) {
		step_frame(dir * 5);
	} else {
		step_frame(dir);
	}

	// if (sk::input_key(sk::key_d) & sk::button_state_just_inactive) {
	// 	step_frame(1);
	// } else if ((sk::input_key(sk::key_a) & sk::button_state_just_inactive)) {
	// 	step_frame(-1);
	// }
}

float
whatever(const hand_bbox_t &past, const hand_bbox_t &present)
{
	return sqrt(pow(past.cx - present.cx, 2) + pow(past.cy - present.cy, 2));
}

void
count_num_confirmed()
{
	st.num_confirmed = 0;
	for (size_t frame_idx = 0; frame_idx < st.frames.size(); frame_idx++) {
		if (st.frames[frame_idx].positions_confirmed) {
			st.num_confirmed++;
		}
	}
}


void
handedness_automata()
{
	struct
	{
		int num_hands = 0;
		int index_first = 0;
	} last_frame[2];

	for (size_t frame_idx = 0; frame_idx < st.frames.size(); frame_idx++) {
		// printf("%zu - ", frame_idx);s
		one_frame_t &frame = st.frames[frame_idx];
		one_frame_t &last_frame_frame_t = st.frames[frame_idx - 1];
		float confidences[2] = {0.0, 0.0};

		for (int view = 0; view < 2; view++) {
			std::vector<hand_bbox_t> &h = frame.views[view].hands;
			std::vector<hand_bbox_t> &h_last = last_frame_frame_t.views[view].hands;
			if (!frame.handedness_keyframe) {
				// If last frame didn't have any hands but this one does (and isn't handedness keyframe), randomly assign handednesses to this frame
				// Shouldn't happen after first run
				if (last_frame[view].num_hands == 0) {
					int cls = 0;
					for (auto &hand : h) {
						// Ugh, rethink this.
						// if (hand.type != REDACTED) {
						hand.type = (enum hand_class)cls;
						cls++;
						// }
					}
				} else {
					std::vector<bool> last_frame_useds;
					std::vector<bool> this_frame_useds;

					std::vector<size_t> last_frame_indices;
					std::vector<size_t> this_frame_indices;
					std::vector<float> dontuse;


					// Strategy here is: We have a big list of palms. Match 'em up to previous palms.
					naive_sort_permutation_by_error<hand_bbox_t, hand_bbox_t>(h_last, h,

					                                                          // bools
					                                                          last_frame_useds, this_frame_useds,

					                                                          last_frame_indices, this_frame_indices, dontuse, whatever);
					bool classes_used[NUM_VALID_HAND_CLASSES] = {}; // {false, false};
					for (size_t i = 0; i < this_frame_indices.size(); i++) {

						h[this_frame_indices[i]].type = h_last[last_frame_indices[i]].type;

						if (h[this_frame_indices[i]].type != UNKNOWN) {
							// if ((h[this_frame_indices[i]].type == EGO_LEFT) || (h[this_frame_indices[i]].type == EGO_RIGHT)) {
							classes_used[h[this_frame_indices[i]].type] = true;
						}
						// printf("%d %zu, %d %zu. ", h_last[last_frame_indices[i]].type, last_frame_indices[i], h[this_frame_indices[i]].type, this_frame_indices[i]);
					}

					// printf(" %zu! \n", this_frame_useds.size());
					for (size_t bbox_idx = 0; bbox_idx < this_frame_useds.size(); bbox_idx++) {
						if (this_frame_useds[bbox_idx] == true) {
							continue;
						}
						h[bbox_idx].type = UNKNOWN;
						for (int c_i = 0; c_i < NUM_VALID_HAND_CLASSES; c_i++) {
							if (classes_used[c_i] == false) {
								classes_used[c_i] = true;
								h[bbox_idx].type = (enum hand_class)c_i;
								break;
							}
						}
					}
				}
			}

			last_frame[view].num_hands = h.size();
			if (last_frame[view].num_hands != 0) {
				last_frame[view].index_first = frame.views[view].hands[0].type;
			}
		}
		// printf("\n");
	}
}

static float
overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;

	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;

	return right - left;
}

static float
boxIntersection(const hand_bbox_t &a, const hand_bbox_t &b)
{
	float w = overlap(a.cx, a.w, b.cx, b.w);
	float h = overlap(a.cy, a.h, b.cy, b.h);

	if (w < 0 || h < 0)
		return 0;

	return w * h;
}

static float
boxUnion(const hand_bbox_t &a, const hand_bbox_t &b)
{
	return a.w * a.h + b.w * b.h - boxIntersection(a, b);
}

float
box_iou(hand_bbox_t &a, hand_bbox_t &b)
{
	return boxIntersection(a, b) / boxUnion(a, b);
}

void
add_or_replace_hand(hand_bbox_t &hand_new, int frame_idx, int focus)
{
	std::vector<hand_bbox_t> &hands = st.frames[frame_idx].views[focus].hands;
	size_t curr_idx = 0;
	while (true) {
		if (curr_idx == hands.size()) {
			break;
		}
		hand_bbox_t &hand = hands[curr_idx];

		printf("add_or_replace: idx %zu size %zu\n", curr_idx, hands.size());

		// If there's a hand in this frame on this side, remove it - it'll overlap with the one we add down there
		if (box_iou(hand, hand_new) > 0.3f) {
			hands.erase(hands.begin() + curr_idx);
			continue;
		}
		curr_idx++;
	}


	st.frames[frame_idx].views[focus].hands.push_back(hand_new);
}

void
link_two_bboxes()
{
	if (st.linking_two_boxes.start_frame_idx == st.linking_two_boxes.end_frame_idx) {
		return;
	}
	if (!(st.linking_two_boxes.start_frame_idx < st.linking_two_boxes.end_frame_idx)) {
		printf("No!\n");
		return;
	}

	hand_bbox_t start = st.frames[st.linking_two_boxes.start_frame_idx].views[st.focus].hands[st.linking_two_boxes.start_frame_bbox_idx];
	hand_bbox_t end = st.frames[st.linking_two_boxes.end_frame_idx].views[st.focus].hands[st.linking_two_boxes.end_frame_bbox_idx];

	for (int i = st.linking_two_boxes.start_frame_idx + 1; i < st.linking_two_boxes.end_frame_idx; i++) {
		double a = math_map_ranges(i, st.linking_two_boxes.start_frame_idx, st.linking_two_boxes.end_frame_idx, 0, 1);

		hand_bbox_t bb = {};
		bb.cx = math_lerp(start.cx, end.cx, a);
		bb.cy = math_lerp(start.cy, end.cy, a);
		bb.w = math_lerp(start.w, end.w, a);
		bb.h = math_lerp(start.h, end.h, a);
		bb.type = UNKNOWN;

		add_or_replace_hand(bb, i, st.focus);
	}
	handedness_automata();
}

void
link_all_bboxes_to_next_confirmed()
{
	int start_idx = st.curr_frame_idx;
	int end_idx = start_idx;
	bool found = false;


	// Try to find a frame after this one that has confirmed positions.
	// If there isn't one, don't set found to true, and just exit + do nothing.
	while (true) {
		end_idx++;
		if (end_idx == st.num_frames) {
			break;
		}
		if (st.frames[end_idx].positions_confirmed) {
			found = true;
			break;
		}
	}
	if (!found) {
		return;
	}

	for (int view = 0; view < 2; view++) {

		// Remove all hands in intervening frames, and confirm next positions.

		for (int frame_idx = start_idx + 1; frame_idx < end_idx; frame_idx++) {
			st.frames[frame_idx].views[view].hands.clear();
			st.frames[frame_idx].positions_confirmed = true;
			st.frames[frame_idx].handedness_keyframe = false;
		}

		std::vector<hand_bbox_t> &start_frame = st.frames[start_idx].views[view].hands;
		std::vector<hand_bbox_t> &end_frame = st.frames[end_idx].views[view].hands;

		std::vector<bool> start_frame_useds;
		std::vector<bool> end_frame_useds;

		std::vector<size_t> start_frame_indices;
		std::vector<size_t> end_frame_indices;
		std::vector<float> dontuse;


		// Strategy here is: We have a big list of palms. Match 'em up to previous palms.
		naive_sort_permutation_by_error<hand_bbox_t, hand_bbox_t>(start_frame, end_frame,

		                                                          // bools
		                                                          start_frame_useds, end_frame_useds,

		                                                          start_frame_indices, end_frame_indices, dontuse, whatever);
		for (size_t idx_idx = 0; idx_idx < end_frame_indices.size(); idx_idx++) {
			hand_bbox_t &start = start_frame[start_frame_indices[idx_idx]];
			hand_bbox_t &end = end_frame[end_frame_indices[idx_idx]];
			for (int frame_idx = start_idx + 1; frame_idx < end_idx; frame_idx++) {
				double a = math_map_ranges(frame_idx, start_idx, end_idx, 0, 1);
				hand_bbox_t bb = {};
				bb.cx = math_lerp(start.cx, end.cx, a);
				bb.cy = math_lerp(start.cy, end.cy, a);
				bb.w = math_lerp(start.w, end.w, a);
				bb.h = math_lerp(start.h, end.h, a);
				bb.type = UNKNOWN;
				st.frames[frame_idx].views[view].hands.push_back(bb);
			}
		}
	}

	step_frame(end_idx - start_idx);

	handedness_automata();
}

void
write_out_json()
{
	cJSON *root = cJSON_CreateObject();
	cJSON *frames = cJSON_AddArrayToObject(root, "frames");
	for (int frame_idx = 0; frame_idx < st.num_frames; frame_idx++) {
		cJSON *frame = cJSON_CreateObject(); // cJSON_AddItemToArray;
		cJSON_AddBoolToObject(frame, "position_confirmed", st.frames[frame_idx].positions_confirmed);
		cJSON_AddBoolToObject(frame, "handedness_keyframe", st.frames[frame_idx].handedness_keyframe);
		cJSON_AddNumberToObject(frame, "timestamp", st.frames[frame_idx].timestamp);
		for (int view = 0; view < 2; view++) {
			cJSON *view_json = cJSON_AddObjectToObject(frame, view_keys[view]);
			cJSON_AddStringToObject(view_json, "filename", st.frames[frame_idx].views[view].filename);

			cJSON *hands = cJSON_AddArrayToObject(view_json, "hands");
			for (auto &hand_v : st.frames[frame_idx].views[view].hands) {
				cJSON *hand_j = cJSON_CreateArray();
				cJSON_AddItemToArray(hand_j, cJSON_CreateNumber(hand_v.cx));
				cJSON_AddItemToArray(hand_j, cJSON_CreateNumber(hand_v.cy));
				cJSON_AddItemToArray(hand_j, cJSON_CreateNumber(hand_v.w));
				cJSON_AddItemToArray(hand_j, cJSON_CreateNumber(hand_v.h));
				cJSON_AddItemToArray(hand_j, cJSON_CreateNumber(hand_v.type));
				cJSON_AddItemToArray(hands, hand_j);
			}
		}
		cJSON_AddItemToArray(frames, frame);
	}
	char *h = cJSON_PrintUnformatted(root);
	FILE *f = fopen((st.paths.root / st.paths.human_annotated).c_str(), "w+");
	fprintf(f, "%s\n", h);
	fflush(f);
	fclose(f);
	free(h);
}

void
update()
{
	// Order doesn't matter much here; it's hard to resize the window AND move the view at the same time
	update_global_mouse();
	manage_screen_resize();
	manage_pan_zoom();
	draw_bound();
	step_frames();
	viz_frames(st);

	// If we're not drawing a new hand, update focus
	if (!(st.drawing_new_frame.active || st.linking_two_boxes.active)) {
		if (sk::input_key(key_i) & sk::button_state_just_inactive) {
			link_all_bboxes_to_next_confirmed();
		}
		if (st.global_mouse_position.x < 0.0f) {
			st.focus = 0;
		} else {
			st.focus = 1;
		}
	}

	if (sk::input_key(sk::key_o)) {
		st.this_frame->positions_confirmed = false;
	} else if ((sk::input_key(sk::key_p))) {
		st.this_frame->positions_confirmed = true;
	}



	const char *p_string;
	const char *h_string;

	if (st.this_frame->positions_confirmed) {
		p_string = "Positions confirmed!";
	} else {
		p_string = "Positions not confirmed!";
	}

	if (st.this_frame->handedness_keyframe) {
		h_string = "Handedness Keyframe!";
	} else {
		h_string = "Not Handedness Keyframe!";
	}

	// Inefficient, too bad.
	count_num_confirmed();

	char guy[1024];
	sprintf(guy, "On frame %d/%d | %d/%d confirmed | %s | %s", st.curr_frame_idx, st.num_frames, st.num_confirmed, st.num_frames, p_string, h_string);
	text(guy, {-1, .4, 0});



	float width = 1.0f;
	float aspect_ratio = 1280.0 / 800.0;
	// 0.5 meters in front of your head's starting position
	// Width, height, 1
	sk::vec3 scale = {width, -width / aspect_ratio, 1};



	sk::matrix px_coord_transforms[2] = {matrix_trs({-width, 0.5f * (width / aspect_ratio), -1}, sk::quat_identity, sk::vec3{1, -1, 1} * (1.0 / 1280.0)),
	                                     matrix_trs({0, 0.5f * (width / aspect_ratio), -1}, sk::quat_identity, sk::vec3{1, -1, 1} * (1.0 / 1280.0))};

	for (int view = 0; view < 2; view++) {
		sk::vec3 offset = {640, 400, 0};
		hierarchy_push(px_coord_transforms[view]);
		sk::vec3 mouse_position;
		mouse_position = hierarchy_to_local_point({st.global_mouse_position.x, st.global_mouse_position.y, 0});

		if (st.focus == view) {
			draw_rectangle_cs({offset.x, offset.y}, {1278, 798}, .2, .006, {30, 255, 20, 128});
			if (!st.drawing_new_frame.active && sk::input_key(sk::key_mouse_right) & sk::button_state_active) {
				remove_hands(&st.this_frame->views[view], mouse_position);
			}
			if (sk::input_key(sk::key_mouse_left) & sk::button_state_just_active) {
				st.drawing_new_frame.active = true;
				st.drawing_new_frame.start_point = mouse_position;
				st.this_frame->views[view].hands.push_back({});
				// if (sk::input_key(sk::key_x)) {
				// 	st.this_frame->views[view].hands.back().type = REDACTED;
				// 	st.this_frame->handedness_keyframe = true;
				// } else {
				st.this_frame->views[view].hands.back().type = UNKNOWN;
				st.this_frame->handedness_keyframe = false;
				// }
			}
			if (!st.linking_two_boxes.active && sk::input_key(sk::key_l) & sk::button_state_just_inactive) {
				size_t i = 0;
				for (; i < st.this_frame->views[view].hands.size(); i++) {
					if (mouse_in_bbox(st.this_frame->views[view].hands[i], mouse_position)) {
						// Just pick the first one. If our mouse is in two bounding boxes, what were we thinking?
						st.linking_two_boxes.active = true;
						st.linking_two_boxes.after_one_frame = true;
						st.linking_two_boxes.start_frame_idx = st.curr_frame_idx;
						st.linking_two_boxes.start_frame_bbox_idx = i;
						break;
					}
				}
			}
			if (st.linking_two_boxes.active) {
				text("Linking!", {900, -40, .2}, 1280, true);
				// printf("%d %d", st.linking_two_boxes.start_frame_idx);
				draw_hand_box(st.frames[st.linking_two_boxes.start_frame_idx].views[view].hands[st.linking_two_boxes.start_frame_bbox_idx]);

				if (!st.linking_two_boxes.after_one_frame && sk::input_key(sk::key_l) & sk::button_state_just_inactive) {
					// Ending the thing.

					size_t i = 0;
					for (; i < st.this_frame->views[view].hands.size(); i++) {
						if (mouse_in_bbox(st.this_frame->views[view].hands[i], mouse_position)) {
							// Just pick the first one. If our mouse is in two bounding boxes, what were we thinking?
							st.linking_two_boxes.end_frame_idx = st.curr_frame_idx;
							st.linking_two_boxes.end_frame_bbox_idx = i;
							link_two_bboxes();
							st.linking_two_boxes.active = false;
							break;
						}
					}
				} else if (sk::input_key(sk::key_esc) & sk::button_state_just_inactive) {
					st.linking_two_boxes.active = false;
				}
				st.linking_two_boxes.after_one_frame = false;
			}
			if (st.drawing_new_frame.active) {
				hand_bbox_t *e = &st.this_frame->views[view].hands.back();

				e->cx = (mouse_position.x + st.drawing_new_frame.start_point.x) / 2;
				e->cy = (mouse_position.y + st.drawing_new_frame.start_point.y) / 2;

				e->w = fabs(mouse_position.x - st.drawing_new_frame.start_point.x);
				e->h = fabs(mouse_position.y - st.drawing_new_frame.start_point.y);
			}
			if (sk::input_key(sk::key_mouse_left) & sk::button_state_just_inactive) {
				st.drawing_new_frame.active = false;
				remove_misclicks(&st.this_frame->views[view]);
				handedness_automata();
			}
			if (sk::input_key(sk::key_s) & sk::button_state_just_inactive) {
				for (auto &hand : st.this_frame->views[view].hands) {
					if (hand.type == EGO_LEFT) {
						hand.type = EGO_RIGHT;
					} else if (hand.type == EGO_RIGHT) {
						hand.type = EGO_LEFT;
					}
				}
				st.this_frame->handedness_keyframe = true;
				handedness_automata();
			}
		}
		sk::render_add_model(st.view[view].img_model, matrix_trs(offset, sk::quat_identity, scale * 1280));
		draw_hands(st.this_frame->views[view].hands);

		std::ostringstream hi;
		hi << st.this_frame->views[view].hands.size() << " hands | " << st.this_frame->timestamp;
		text(hi.str().c_str(), {0, -40, .2}, 1280, true);

		hierarchy_pop();
	}

	if (input_key(sk::key_w) & sk::button_state_just_inactive) {
		printf("Saving file!\n");
		write_out_json();
	}



	sk::vec3 ax_pose = {};
	ax_pose.x = st.global_mouse_position.x;
	ax_pose.y = st.global_mouse_position.y;
	ax_pose.z = -.2f;
	draw_cross(ax_pose);
}


void
read_json(cJSON *array)
{
	cJSON *frame;
	cJSON_ArrayForEach(frame, array)
	{

		one_frame_t this_frame;
		u_json_get_bool(u_json_get(frame, "position_confirmed"), &this_frame.positions_confirmed);
		u_json_get_bool(u_json_get(frame, "handedness_keyframe"), &this_frame.handedness_keyframe);
		const cJSON *time = u_json_get(frame, "timestamp");
		uint64_t ts = time->valuedouble;
		this_frame.timestamp = ts;

		for (int view = 0; view < 2; view++) {
			const cJSON *side = u_json_get(frame, view_keys[view]);
			this_frame.views[view].filename = cJSON_GetStringValue(u_json_get(side, "filename"));

			cJSON *hand;
			cJSON_ArrayForEach(hand, u_json_get(side, "hands"))
			{
				float his[4];
				int handedness;
				// int annotated_machine_or_human;
				u_json_get_float_array(hand, his, 4); // First four; cx, cy, w, h
				u_json_get_int(cJSON_GetArrayItem(hand, 4), &handedness);
				// u_json_get_int(cJSON_GetArrayItem(hand, 5), &annotated_machine_or_human);

				hand_bbox_t bb;
				bb.cx = his[0];
				bb.cy = his[1];
				bb.w = his[2];
				bb.h = his[3];
				bb.type = (enum hand_class)handedness;
				// bb.annotated_machine_or_human = annotated
				this_frame.views[view].hands.push_back(bb);
			}
		}
		st.frames.push_back(this_frame);
	}
}



cJSON *
open_file(const char *filename)
{
	FILE *f = fopen(filename, "r");
	if (f == NULL) {
		return NULL;
	}
	char *dat = u_file_read_content(f);
	fclose(f);

	cJSON *h = cJSON_Parse(dat);
	free(dat);
	return h;
}



int
main(int argc, char **argv)
{
	CLI::App app{"Bounding box annotator!!"};

	std::string euroc_path;
	app.add_option("--euroc_path", euroc_path, "Path to EuRoC dataset")->required(true);

	CLI11_PARSE(app, argc, argv);

	std::cout << euroc_path << std::endl;

	st.paths.root = euroc_path;

	cJSON *machine = open_file((st.paths.root / st.paths.machine_annotated).c_str());
	cJSON *human = open_file((st.paths.root / st.paths.human_annotated).c_str());

	st.json_root = human;
	if (st.json_root == NULL) {
		st.json_root = machine;
		if (st.json_root == NULL) {
			U_LOG_E("EuRoC dataset does not contain a `machine_annotated.json` or a `human_annotated.json`!");
			U_LOG_E("This is probably just a dataset that you never ran the auto-annotator script on.");
			U_LOG_E("Run `py/run_machine_annotation.py --euroc_path %s`, then come back!", euroc_path.c_str());
			exit(1);
		}
	}


	sk_settings_t settings = {};
	settings.app_name = "bounding box annotator!";
	settings.display_preference = sk::display_mode_flatscreen;
	settings.disable_flatscreen_mr_sim = true;
	if (!sk_init(settings))
		return 1;

	sk::mesh_t m = sk::mesh_gen_plane({1, 1}, {0, 0, 1}, {0, 1, 0});

	for (int i = 0; i < 2; i++) {
		st.view[i].img_tex = tex_create(tex_type_image, sk::tex_format_rgba128);
		tex_set_sample(st.view[i].img_tex, sk::tex_sample_point);


		st.view[i].img_material = material_copy_id("default/material");

		material_set_float(st.view[i].img_material, "tex_scale", 1);
		material_set_cull(st.view[i].img_material, sk::cull_none);

		material_set_texture(st.view[i].img_material, "diffuse", st.view[i].img_tex);

		st.view[i].img_model = model_create_mesh(m, st.view[i].img_material);
	}



	st.json_frames = cJSON_GetObjectItemCaseSensitive(st.json_root, "frames");
	read_json(st.json_frames);
	handedness_automata();
	st.num_frames = cJSON_GetArraySize(st.json_frames);

	// step_frame(200);
	step_frame(71);

	styles[0] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {1, 0, 1, .7});
	styles[1] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {0, 1, 1, .7});
	styles[2] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {1, 0, 0, .7});



	sk::render_set_projection(projection_ortho);
	sk::render_enable_skytex(false);



	while (sk_step(update)) {
		// Do nothing! update did everything already!
	};

	sk_shutdown();
	return 0;
}
