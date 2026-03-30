#include "pose_diversity_hand_maker.h"
#include "util/u_misc.h"
#include "xrt/xrt_defines.h"
#include "math/m_vec3.h"
#include "make_hand_proportions.hpp"

#include "pose_csv.hpp"
#include "kine_lm/lm_interface.hpp"
#include "kine_lm/lm_defines.hpp"
#include <random>
#include <iostream>

#define printf_pose(pose)                                                                                              \
	printf("%f %f %f  %f %f %f %f\n", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,       \
	       pose.orientation.y, pose.orientation.z, pose.orientation.w);


namespace lm = xrt::tracking::hand::mercury::lm;
float deg2rad = M_PI / 180.f;

struct fingerpose_creator
{
	bool use_real_finger_data;
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

	TrajectoryReader<26> *finger_pose_reader;
	uint64_t fp_idx = 0;
};


float radius = deg2rad * 30.0f;

float curled = deg2rad * -270.0f;
float uncurled = 30.0f * deg2rad;

// Honestly can't do much more than 90 with my hand
float curled_thumb = -90.0f * deg2rad;
float uncurled_thumb = 60.0f * deg2rad;

float factor = 0.01;
float factor_r = 0.01;

float
normalf(fingerpose_creator &st, float center, float stddev)
{
	std::normal_distribution<float>::param_type param{center, stddev};
	return fmax(0, st.ndist(st.gen, param));
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
middle_finger(fingerpose_creator &st, lm::optimizer_input *optinp)
{
	optinp->curls.c[0].value = uniform(st, curled / 2, radius * 5);
	optinp->curls.c[1].value = uniform(st, curled, radius);
	optinp->curls.c[2].value = uniform(st, uncurled, radius);
	optinp->curls.c[3].value = uniform(st, curled, radius);
	optinp->curls.c[4].value = uniform(st, curled, radius);

	optinp->curls.c[0].factor = uniformf(st, factor, factor_r * 0.1);
	optinp->curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[4].factor = uniformf(st, factor, factor_r);
}

void
spoder(fingerpose_creator &st, lm::optimizer_input *optinp)
{
	optinp->curls.c[0].value = uniform(st, curled / 2, radius * 5);
	optinp->curls.c[1].value = uniform(st, uncurled, radius);
	optinp->curls.c[2].value = uniform(st, curled, radius);
	optinp->curls.c[3].value = uniform(st, curled, radius);
	optinp->curls.c[4].value = uniform(st, uncurled, radius);

	optinp->curls.c[0].factor = uniformf(st, factor, factor_r * 0.1);
	optinp->curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[4].factor = uniformf(st, factor, factor_r);
}

void
fingergun(fingerpose_creator &st, lm::optimizer_input *optinp)
{
	optinp->curls.c[0].value = uniform(st, uncurled, radius);
	optinp->curls.c[1].value = uniform(st, uncurled, radius);
	optinp->curls.c[2].value = uniform(st, curled, radius);
	optinp->curls.c[3].value = uniform(st, curled, radius);
	optinp->curls.c[4].value = uniform(st, curled, radius);

	optinp->curls.c[0].factor = uniformf(st, factor, factor_r * 0.3);
	optinp->curls.c[1].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[2].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[3].factor = uniformf(st, factor, factor_r);
	optinp->curls.c[4].factor = uniformf(st, factor, factor_r);
}

#define if_01(stuff)                                                                                                   \
	do {                                                                                                           \
		if (uniformtb(st, 0, 1) < 0.1) {                                                                       \
			stuff                                                                                          \
		}                                                                                                      \
	} while (0)

#define if_02 if (uniformtb(st, 0, 1) < 0.05)

void
do_it(fingerpose_creator &st, lm::optimizer_input *optinp)
{
	for (int i = 0; i < 5; i++) {
		if_02
		{
			optinp->curls.c[i].value = uniformtb(st, uncurled, curled);
		}
		if_02
		{
			optinp->curls.c[i].factor = normalf(st, 0, 1.0);
		}

		if_02
		{
			optinp->splays.s[i].value = uniformtb(st, -40 * deg2rad, 40 * deg2rad);
		}
		if_02
		{
			optinp->splays.s[i].factor = normalf(st, 0, 1.0);
		}
	}

	if_02
	{
		optinp->curls.c[0].value = uniformtb(st, uncurled_thumb, curled_thumb);
	}


	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		if_02
		{
			if_02
			{
				optinp->tip_touches.t[i].value = normalf(st, 1.0, 0.2);
			}
			else
			{
				optinp->tip_touches.t[i].value = normalf(st, 0, 0.05);
			}
			optinp->tip_touches.t[i].factor = normalf(st, 0, 1.0);
		}
	}
}

// This is duplicated code. need a template but ehhhh
void
random_init(fingerpose_creator &st, lm::optimizer_input *optinp)
{
	for (int i = 0; i < 5; i++) {
		optinp->curls.c[i].value = uniformtb(st, uncurled, curled);
		optinp->curls.c[i].factor =
		    normalf(st, 0.4, 0.6); // Can't be 0 - then it won't move at all, but let's be weird about it
		optinp->splays.s[i].value = uniformtb(st, -40 * deg2rad, 40 * deg2rad);
		optinp->splays.s[i].factor = normalf(st, 0.4, 0.6);
	}

	{
		// Overwrites the old value from the old loop. /shrug.
		optinp->curls.c[0].value = uniformtb(st, uncurled_thumb, curled_thumb);
	}

#if 0
	// Actually we don't want a whole lot of tip touches
	for (int i = 0; i < lm::NUM_TIP_TOUCH_COMBOS; i++) {
		if_02
		{
			if_02
			{
				optinp->tip_touches.t[i].value = normalf(st, 1.0, 0.2);
			}
			else
			{
				optinp->tip_touches.t[i].value = normalf(st, 0, 0.05);
			}
			optinp->tip_touches.t[i].factor = normalf(st, 0, 1.0);
		}
	}
#endif
}


void
finger_pose_step(fingerpose_creator *cr, xrt_hand_joint_set *out_optimized_hand, TrajectorySample<26> *out_sample_used)
{
	fingerpose_creator &st = *cr;

	// if (sk::input_key(sk::key_f) & sk::button_state_active) {


	std::random_device dev;

	auto mt = std::mt19937(dev());
	auto rd = std::uniform_real_distribution<float>(0, 1);

	if (st.idx == 0) {
		random_init(st, &st.optinp);
	} else if ((st.idx % 1) == 0) {
		do_it(st, &st.optinp);
	}

	st.idx++;


	TrajectorySample<26> samp = {};

	// Here
	if (st.use_real_finger_data) {
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
create_finger_pose(TrajectoryReader<26> *finger_mocap, const char *proportions_json, int64_t start_ts, float frametime)
{
	struct fingerpose_creator *cr = new fingerpose_creator;
	fingerpose_creator &st = *cr;
	st.fp_idx = start_ts;

	st.finger_pose_reader = finger_mocap;

	st.frametime = frametime;

	st.use_real_finger_data = (finger_mocap != NULL);

	lm::hand_proportions proportions = {};
	lm::HandLimit limit = {};
	make_hand_proportions(proportions_json, proportions, limit);

	st.hp = proportions;

	struct xrt_pose bleh = {};
	bleh.orientation.w = 1.0;
	bleh.position.x = 0.5;

	lm::optimizer_create(bleh, 0, U_LOGGING_ERROR, proportions, limit, &st.lm);

	U_LOG_E("Created! Frametime was %f, use_real_data was %d", st.frametime, st.use_real_finger_data);

	return cr;
}

void
finger_pose_delete(fingerpose_creator **cr)
{
	delete *cr;
	*cr = NULL;
}