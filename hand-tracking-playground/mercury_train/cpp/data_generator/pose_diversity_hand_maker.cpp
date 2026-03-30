#include "pose_diversity_hand_maker.h"
#include "aux/u_random_distribution.h"
#include "util/u_misc.h"
#include "xrt/xrt_defines.h"
#include "math/m_vec3.h"
#include "make_hand_proportions.hpp"

#include "aux/pose_csv.hpp"
#include "kine_lm/lm_interface.hpp"
#include "kine_lm/lm_defines.hpp"
#include <random>
#include <iostream>

#define printf_pose(pose)                                                                                              \
	printf("%f %f %f  %f %f %f %f\n", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,       \
	       pose.orientation.y, pose.orientation.z, pose.orientation.w);


namespace lm = xrt::tracking::hand::mercury::lm;
float deg2rad = M_PI / 180.f;


enum fingerpose_creator_run_type
{
	REAL_DATA,
	RANDOM_PLUS_CONTRIVED_DATA,
	RANDOM_DATA
};

enum hcp
{
	RANDOM,
	MIDDLE_FINGER,
	OK_HAND,
	INDEX_TOUCH_THUMB, // This is the same as ok_hand but doesn't emphasize uncurled other-fingers.
	MIDDLE_TOUCH_THUMB,
	RING_TOUCH_THUMB,
	LITTLE_TOUCH_THUMB,
	POINT_ALL_FINGERS_CLOSED,
	SPIDERMAN,
	// WOW_MOM,
	BALLED_FIST,
	CALL_ME
};

hcp hello[] = {
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
    MIDDLE_FINGER,            //
                              //
    OK_HAND,                  //
    OK_HAND,                  //
                              //
    INDEX_TOUCH_THUMB,        //
    INDEX_TOUCH_THUMB,        //
    INDEX_TOUCH_THUMB,        //
    INDEX_TOUCH_THUMB,        //
                              //
    MIDDLE_TOUCH_THUMB,       //
    MIDDLE_TOUCH_THUMB,       //
    MIDDLE_TOUCH_THUMB,       //
                              //
    RING_TOUCH_THUMB,         //
    RING_TOUCH_THUMB,         //
    RING_TOUCH_THUMB,         //
                              //
    LITTLE_TOUCH_THUMB,       //
    LITTLE_TOUCH_THUMB,       //
    LITTLE_TOUCH_THUMB,       //
                              //
    POINT_ALL_FINGERS_CLOSED, //
    POINT_ALL_FINGERS_CLOSED, //
                              //
    SPIDERMAN,                //
    SPIDERMAN,                //
                              //
    BALLED_FIST,              //
    BALLED_FIST,              //
    BALLED_FIST,              //
    BALLED_FIST,              //
                              //
    CALL_ME,                  //
    CALL_ME,                  //
    CALL_ME,                  //
};

struct fingerpose_creator
{
	enum fingerpose_creator_run_type run_type;
	struct lm::KinematicHandLM *lm = nullptr;
	struct xrt_hand_joint_set set;

	lm::optimizer_input optinp = {};

	int idx = 0;
	std::random_device random_device;
	std::mt19937 gen;
	std::normal_distribution<float> ndist;
	std::uniform_real_distribution<float> udist;

	lm::hand_proportions hp;
	float frametime;

	hcp current_status;

	TrajectoryReader<26> *finger_pose_reader;
	uint64_t fp_idx = 0;
};


float radius = deg2rad * 30.0f;

float curled = deg2rad * -270.0f;
float uncurled = 30.0f * deg2rad;

float half_curled = (curled + uncurled) / 2;

float super_curled = deg2rad * -310.0f;
float meh_uncurled = 5.0f * deg2rad;

// Honestly can't do much more than 90 with my hand
// moshi, jan 2: what? no the thumb X range of motion is like at least 180 degrees
// hmm, a
float curled_thumb = -130.0f * deg2rad;
float uncurled_thumb = 60.0f * deg2rad;

float factor = .3;
float factor_r = 0.01;

#define if_02 if (uniformtb(st, 0, 1) < 0.05)

#define if_p(val) if (uniformtb(st, 0, 1) < val)

float
normalf(fingerpose_creator &st, float center, float stddev)
{
	std::normal_distribution<float>::param_type param{center, stddev};
	return fmax(0, st.ndist(st.gen, param));
}
float
normalabs(fingerpose_creator &st, float center, float stddev)
{
	std::normal_distribution<float>::param_type param{center, stddev};
	return fabsf(st.ndist(st.gen, param));
}

float
uniformtb(fingerpose_creator &st, float bottom, float top)
{
	std::uniform_real_distribution<float>::param_type param{bottom, top};
	return st.udist(st.gen, param);
}

float
uniform(fingerpose_creator &st, float center, float radius)
{
	std::uniform_real_distribution<float>::param_type param{center - radius, center + radius};
	return st.udist(st.gen, param);
}

float
uniformf(fingerpose_creator &st, float center, float radius)
{

	std::uniform_real_distribution<float>::param_type param(center - radius, center + radius);
	return fmax(0, st.udist(st.gen, param));
}

void
do_it_tip_touches(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		if_02
		{
			if_02
			{
				optinp.tip_touches.t[i].value = normalf(st, 1.0, 0.2);
			}
			else
			{
				optinp.tip_touches.t[i].value = normalf(st, 0, 0.05);
			}
			optinp.tip_touches.t[i].factor = normalf(st, 0, 1.0);
		}
	}
}

void
do_it_curl_one_finger(fingerpose_creator &st, lm::optimizer_input &optinp, int i, float top, float bottom)
{
	if_02
	{
		optinp.curls.c[i].value = uniformtb(st, top, bottom);
	}
	if_02
	{
		optinp.curls.c[i].factor = normalf(st, 0, 0.8);
	}
}

void
do_it_splay_one_finger(fingerpose_creator &st, lm::optimizer_input &optinp, int i)
{
	if_02
	{
		optinp.splays.s[i].value = uniformtb(st, -40 * deg2rad, 40 * deg2rad);
	}
	if_02
	{
		optinp.splays.s[i].factor = normalf(st, 0, 0.8);
	}
}



// This is duplicated code. need a template but ehhhh
void
initial_pose(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	for (int i = 0; i < 5; i++) {
		optinp.curls.c[i].value = uniformtb(st, uncurled, curled);
		optinp.curls.c[i].factor =
		    normalf(st, 0.4, 0.6); // Can't be 0 - then it won't move at all, but let's be weird about it
		optinp.splays.s[i].value = uniformtb(st, -40 * deg2rad, 40 * deg2rad);
		optinp.splays.s[i].factor = normalf(st, 0.4, 0.6);
	}

	{
		// Overwrites the old value from the old loop. /shrug.
		optinp.curls.c[0].value = uniformtb(st, uncurled_thumb, curled_thumb);
	}


	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		optinp.tip_touches.t[i].factor = 0.0f;
	}
}

void
fisbys_sweep(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	for (int i = 0; i < 5; i++) {
		optinp.curls.c[i].factor *= 0.5f;
		optinp.splays.s[i].factor *= 0.5f;

		// don't do this for thumb
		if (i == 0) {
			continue;
		}

		if_p(0.5f)
		{
			optinp.curls.c[i].value = uniformtb(st, uncurled, curled);
		}
		if_p(0.5f)
		{
			optinp.splays.s[i].value = uniformtb(st, -40 * deg2rad, 40 * deg2rad);
		}
	}

	if_p(0.5f)
	{
		// Overwrites the old value from the old loop. /shrug.
		optinp.curls.c[0].value = uniformtb(st, uncurled_thumb, curled_thumb);
	}


	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		// generally, getting "some amount" of them aw
		if_p(0.3f)
		{
			optinp.tip_touches.t[i].factor = normalabs(st, 0, 1.0);

			optinp.tip_touches.t[i].value = 0.3f; // 0.15m, get them away!
		}
		else
		{
			optinp.tip_touches.t[i].factor = 0;
		}
	}
}

void
random(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	// four fingers
	for (int i = 1; i < 5; i++) {
		do_it_curl_one_finger(st, optinp, i, uncurled, curled);
		do_it_splay_one_finger(st, optinp, i);
	}

	// thumb
	do_it_splay_one_finger(st, optinp, 0);
	do_it_curl_one_finger(st, optinp, 0, uncurled_thumb, curled_thumb);



	do_it_tip_touches(st, optinp);
}

void
finger_touch(fingerpose_creator &st, lm::optimizer_input &optinp, enum lm::tip_touch_combos c)
{
	random(st, optinp);
	optinp.tip_touches.t[c].value = 0.001; // 2mm
	// we should probably have a norm
	optinp.tip_touches.t[c].factor = normalf(st, 10.0f, 0.1f);
	st.optinp.plausibility_factor = 0.5f;

	for (size_t i = 0; i < 4; i++) {
		// If this is the finger the thumb is touching, continue.
		if (i == lm::ttc_elements[c][0]) {
			continue;
		}
		if_p(0.3)
		{
			optinp.curls.c[i].value = uniformtb(st, uncurled, half_curled);
		}
	}
	optinp.no_collide_factor = 0.1f;
	optinp.plausibility_factor = 0.0f;
}

void
middle_finger(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	random(st, optinp);
	optinp.curls.c[1].value = uniform(st, super_curled, radius * 3);
	optinp.curls.c[2].value = uniform(st, meh_uncurled, radius * 3);
	optinp.curls.c[3].value = uniform(st, super_curled, radius * 3);
	optinp.curls.c[4].value = uniform(st, super_curled, radius * 3);

	// optinp.curls.c[0].factor = uniformf(st, factor, factor_r * 0.1);
	optinp.curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[2].factor = uniformf(st, factor * 2.0, factor_r);
	optinp.curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[4].factor = uniformf(st, factor, factor_r);
	st.optinp.plausibility_factor = 0.00f;
}



void
index_touch_thumb(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	finger_touch(st, optinp, lm::IN_TH);
}
void
middle_touch_thumb(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	finger_touch(st, optinp, lm::MI_TH);
}
void
ring_touch_thumb(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	finger_touch(st, optinp, lm::RI_TH);
}
void
little_touch_thumb(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	finger_touch(st, optinp, lm::LI_TH);
}

void
ok_hand(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	finger_touch(st, optinp, lm::IN_TH);
	if_p(0.1)
	{
		optinp.curls.c[1].value = uniform(st, curled, 0.3);
		optinp.curls.c[1].factor = uniformf(st, factor, factor_r);
	}

	for (int i = 2; i < 5; i++) {
		if_p(0.5)
		{
			optinp.curls.c[i].value = uniformtb(st, uncurled, half_curled);
			optinp.curls.c[i].factor = uniformf(st, factor, factor_r);
		}
	}
}

void
point_all_fingers_closed(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	random(st, optinp);
	// Note we're not telling the thumb what to do
	// optinp.curls.c[0].value = uniform(st, curled_thumb*1.5, radius);
	optinp.curls.c[1].value = uniform(st, uncurled, radius);
	optinp.curls.c[2].value = uniform(st, super_curled, radius * 2);
	optinp.curls.c[3].value = uniform(st, super_curled, radius * 2);
	optinp.curls.c[4].value = uniform(st, super_curled, radius * 2);

	// optinp.curls.c[0].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[1].factor = uniformf(st, factor * 2.0, factor_r);
	optinp.curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[4].factor = uniformf(st, factor, factor_r);


	if_p(0.1)
	{
		optinp.tip_touches.t[lm::LI_TH].value = 0.001;
		optinp.tip_touches.t[lm::LI_TH].factor = 10.0;
	}
	if_p(0.1)
	{
		optinp.tip_touches.t[lm::RI_TH].value = 0.001;
		optinp.tip_touches.t[lm::RI_TH].factor = 10.0;
	}
	if_p(0.1)
	{
		optinp.tip_touches.t[lm::MI_TH].value = 0.001;
		optinp.tip_touches.t[lm::MI_TH].factor = 10.0;
	}
	// optinp.tip_touches.t[lm::MI_TH].value = 0.001;

	// }

	optinp.plausibility_factor = 0.0f;
	optinp.no_collide_factor = 0.05f;
}

void
balled_fist(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	random(st, optinp);
	// Note we're not telling the thumb what to do
	optinp.curls.c[0].value = uniform(st, curled_thumb * 1.5, radius);
	optinp.curls.c[1].value = uniform(st, super_curled, radius);
	optinp.curls.c[2].value = uniform(st, super_curled, radius * 2);
	optinp.curls.c[3].value = uniform(st, super_curled, radius * 2);
	optinp.curls.c[4].value = uniform(st, super_curled, radius * 2);

	optinp.curls.c[0].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[4].factor = uniformf(st, factor, factor_r);


	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		if_p(0.95)
		{
			optinp.tip_touches.t[i].value = uniformtb(st, 0.001, 0.03);
			optinp.tip_touches.t[i].factor = uniformtb(st, 2.0, 10.0);
		}
	}

	// optinp.plausibility_factor = 0.0f;
	optinp.no_collide_factor = 0.1f;
}

void
spider_man(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	random(st, optinp);
	// Note we're not telling the thumb what to do
	optinp.curls.c[1].value = uniform(st, uncurled, radius);
	optinp.curls.c[2].value = uniform(st, curled, radius);
	optinp.curls.c[3].value = uniform(st, curled, radius);
	optinp.curls.c[4].value = uniform(st, uncurled, radius);

	optinp.curls.c[1].factor = uniformf(st, factor * 1.6, factor_r);
	optinp.curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[4].factor = uniformf(st, factor * 1.6, factor_r);
	st.optinp.plausibility_factor = 0.0f;
}

void
call_me(fingerpose_creator &st, lm::optimizer_input &optinp)
{
	random(st, optinp);
	optinp.curls.c[0].value = uniform(st, uncurled_thumb - 0.1f, radius * 4);
	optinp.curls.c[1].value = uniform(st, curled, radius);
	optinp.curls.c[2].value = uniform(st, curled, radius);
	optinp.curls.c[3].value = uniform(st, curled, radius);
	optinp.curls.c[4].value = uniform(st, uncurled, radius);

	optinp.curls.c[0].factor = uniformf(st, factor * 1.6, factor_r);
	optinp.curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp.curls.c[4].factor = uniformf(st, factor * 1.6, factor_r);
	st.optinp.plausibility_factor = 0.0f;
}

void
pick_status(fingerpose_creator &st)
{
	struct uniform_distribution *dist = NULL;

	int num_choice_max = ARRAY_SIZE(hello);
	int num_choice = u_random_distribution_get_sample_int64_t(0, num_choice_max);

	assert(num_choice < num_choice_max);

	st.current_status = hello[num_choice];
}

void
do_artificial_pose(fingerpose_creator &st)
{


	// Reset this every time because we're LAZY
	st.optinp.neutral_finger_metacarpals_factor = 1.0f;
	st.optinp.plausibility_factor = 1.0f;
	st.optinp.no_collide_factor = 1.0f;
	st.optinp.stability_factor = 1.0f;

	if (st.run_type == RANDOM_PLUS_CONTRIVED_DATA) {
		int frames_since_last_update = st.idx % 50;
		if (frames_since_last_update == 0) {
			fisbys_sweep(st, st.optinp);


			if_p(0.3f)
			{
				st.current_status = RANDOM;
			}
			else
			{
				pick_status(st);
			}
		}
		if (frames_since_last_update < 3) {
			st.optinp.stability_factor = 0.5f;
			st.optinp.no_collide_factor = 0.6f;
		}
	} else {
		st.current_status = RANDOM;
	}



	switch (st.current_status) {
	case RANDOM: {
		random(st, st.optinp);
		break;
	};
	case MIDDLE_FINGER: {
		middle_finger(st, st.optinp);
		break;
	};
	case OK_HAND: {
		ok_hand(st, st.optinp);
		break;
	};
	case INDEX_TOUCH_THUMB: {
		index_touch_thumb(st, st.optinp);
		break;
	};
	case MIDDLE_TOUCH_THUMB: {
		middle_touch_thumb(st, st.optinp);
		break;
	};
	case RING_TOUCH_THUMB: {
		ring_touch_thumb(st, st.optinp);
		break;
	};
	case LITTLE_TOUCH_THUMB: {
		little_touch_thumb(st, st.optinp);
		break;
	};
	case POINT_ALL_FINGERS_CLOSED: {
		point_all_fingers_closed(st, st.optinp);
		break;
	}
	case SPIDERMAN: {
		spider_man(st, st.optinp);
		break;
	}
	case BALLED_FIST: {
		balled_fist(st, st.optinp);
		break;
	}
	case CALL_ME: {
		call_me(st, st.optinp);
		break;
	}
	default: {
		U_LOG_E("Oh no! Got status %d", st.current_status);
		assert(false);
	}
	}
}



void
finger_pose_step(fingerpose_creator *cr, xrt_hand_joint_set *out_optimized_hand, TrajectorySample<26> *out_sample_used)
{
	fingerpose_creator &st = *cr;

	// if (sk::input_key(sk::key_f) & sk::button_state_active) {


	std::random_device dev;

	auto mt = std::mt19937(dev());
	auto rd = std::uniform_real_distribution<float>(0, 1);


	st.optinp.plausibility_factor = 1.0f;
	if (st.idx == 0) {
		initial_pose(st, st.optinp);

		if (st.run_type == RANDOM_PLUS_CONTRIVED_DATA) {
			pick_status(st);
		} else {
			st.current_status = RANDOM;
		}

	} else {
		do_artificial_pose(st);
	}


	// // middle_finger(st, &st.optinp);
	// ok_hand(st, &st.optinp);
	// // st.optinp.plausibility_factor = 0.0f;
	// st.optinp.neutral_finger_metacarpals_factor = 1;

	st.idx++;


	TrajectorySample<26> samp = {};

	// Here
	if (st.run_type == REAL_DATA) {
		bool retval = st.finger_pose_reader->get_value(st.fp_idx, samp);
		if (!retval) {
			U_LOG_E(
			    "We seeked past the end of the fingerpose dataset! Our fp_idx was %ld, and the last "
			    "timestamp in the dataset was %ld",
			    st.fp_idx, st.finger_pose_reader->values.back().timestamp);
			abort();
		}

		float sample_hand_length =
		    m_vec3_len(samp[XRT_HAND_JOINT_WRIST].position - samp[XRT_HAND_JOINT_MIDDLE_PROXIMAL].position) + //
		    m_vec3_len(samp[XRT_HAND_JOINT_MIDDLE_PROXIMAL].position -
		               samp[XRT_HAND_JOINT_MIDDLE_INTERMEDIATE].position) + //
		    m_vec3_len(samp[XRT_HAND_JOINT_MIDDLE_INTERMEDIATE].position -
		               samp[XRT_HAND_JOINT_MIDDLE_DISTAL].position) + //
		    m_vec3_len(samp[XRT_HAND_JOINT_MIDDLE_DISTAL].position - samp[XRT_HAND_JOINT_MIDDLE_TIP].position);

		float sample_desired_hand_length = 0;

		sample_desired_hand_length += vector_length(st.hp.rel_translations.t[2][0]);
		sample_desired_hand_length += vector_length(st.hp.rel_translations.t[2][1]);
		sample_desired_hand_length += vector_length(st.hp.rel_translations.t[2][2]);
		sample_desired_hand_length += vector_length(st.hp.rel_translations.t[2][3]);
		sample_desired_hand_length += vector_length(st.hp.rel_translations.t[2][4]);

		sample_desired_hand_length *= st.hp.hand_size;

		float mul = sample_desired_hand_length / sample_hand_length;

		for (int i = 0; i < 26; i++) {
			samp[i].position = samp[i].position * mul;
		}



		st.optinp.target_pose = samp;
		st.optinp.use_target_pose = true;
	} else {
		U_ZERO(&st.optinp.target_pose);
		st.optinp.use_target_pose = false;
	}

	lm::optimizer_run(st.lm, st.optinp, st.set);

	st.fp_idx += float(st.frametime) * float(U_TIME_1S_IN_NS);


	*out_optimized_hand = st.set;
	if (out_sample_used != NULL) {
		*out_sample_used = samp;
	}
}

fingerpose_creator *
create_finger_pose(TrajectoryReader<26> *finger_mocap, const cJSON *proportions_json, int64_t start_ts, float frametime)
{
	struct fingerpose_creator *cr = new fingerpose_creator;
	fingerpose_creator &st = *cr;
	st.fp_idx = start_ts;

	st.finger_pose_reader = finger_mocap;

	st.frametime = frametime;

	if (finger_mocap != NULL) {
		st.run_type = REAL_DATA;
	} else {
		if_p(0.4)
		{
			st.run_type = RANDOM_DATA;
		}
		else
		{
			st.run_type = RANDOM_PLUS_CONTRIVED_DATA;
		}
	}

	lm::hand_proportions proportions = {};
	lm::HandLimit limit = {};
	make_hand_proportions(proportions_json, proportions, limit);

	st.hp = proportions;

	struct xrt_pose bleh = {};
	bleh.orientation.w = 1.0;
	bleh.position.x = 0.5;

	lm::optimizer_create(bleh, 0, U_LOGGING_ERROR, proportions, limit, &st.lm);

	U_LOG_E("Created! Frametime was %f, run_type was %d", st.frametime, st.run_type);

	return cr;
}

void
finger_pose_delete(fingerpose_creator **cr)
{
	delete *cr;
	*cr = NULL;
}