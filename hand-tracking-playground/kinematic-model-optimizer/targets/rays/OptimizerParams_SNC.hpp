#include "util/u_logging.h"
#include "math/m_api.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "hand_interface.hpp"
#include "ceres/rotation.h"
#include <cmath>
#include <iostream>

#include "defines.hpp"
#include "tiny_math.hpp"


#define USE_HAND_SIZE
#define USE_HAND_TRANSLATION
#define USE_HAND_ORIENTATION
#define USE_EVERYTHING_ELSE


static constexpr size_t kMetacarpalBoneDim = 3;
static constexpr size_t kProximalBoneDim = 2;
static constexpr size_t kFingerDim = kProximalBoneDim + kMetacarpalBoneDim + 2;
static constexpr size_t kThumbDim = kMetacarpalBoneDim + 2;
static constexpr size_t kHandSizeDim = 1;
static constexpr size_t kHandTranslationDim = 3;
static constexpr size_t kHandOrientationDim = 3;

// static constexpr size_t kHandDim =
//     kHandSizeDim + kHandTranslationDim + kHandOrientationDim + (4 * kFingerDim) + kThumbDim;

// constexpr size_t kFunctorInputDim = 12 + ((3 + 4) * 4);
// All this weirdness so you can comment out just one line at a time
constexpr size_t kFunctorInputDim = //
#ifdef USE_HAND_SIZE                //
    kHandSizeDim +                  //
#endif                              //
#ifdef USE_HAND_TRANSLATION         //
    kHandTranslationDim +           //
#endif                              //
#ifdef USE_HAND_ORIENTATION         //
    kHandOrientationDim +           //
#endif                              //
#ifdef USE_EVERYTHING_ELSE          //
    kThumbDim +                     //
    (kFingerDim * 4) +              //
#endif                              //
    0;                              //

// constexpr size_t kFunctorInputDim = 11 + ((3 + 4) * 4);
// constexpr size_t kFunctorInputDim = 6;
// constexpr size_t kFunctorInputDim = 6+((3 + 4) * 4);

// constexpr size_t kFunctorInputDim = 10;
// constexpr size_t kFunctorInputDim = 7; // hand_size + pos + ori
// constexpr size_t kFunctorInputDim = 6;
// static constexpr size_t kOutputErrorVectorSize = (21 * 3) + 1; // joints, plus diff hand size from last frame.

// hand size. 1

// quat diff rotation from last frame. 1
// diff root bone pose from last frame. 3

// thumb mcp swing twist. 3
// thumb mcp curl. 2

// finger mcp. 3
// finger pxm swing. 2
// finger curls. 2

// orientation diff. 1 https://math.stackexchange.com/a/90098

// position diff. 3


// 1 + 3 + 2 + ((2+2)*4) + 1 + 3

static constexpr size_t kHRTC_HandSize = 1;
static constexpr size_t kHRTC_RootBoneTranslation = 3;
static constexpr size_t kHRTC_RootBoneOrientation = 3; // Direct difference between the two angle-axis rotations. This
                                                       // works well enough because the rotation should be small.

static constexpr size_t kHRTC_ThumbMCPSwingTwist = 3;
static constexpr size_t kHRTC_ThumbCurls = 2;

static constexpr size_t kHRTC_FingerMCPSwingTwist = 3;
static constexpr size_t kHRTC_FingerPXMSwing = 3;
static constexpr size_t kHRTC_FingerCurls = 2;



static constexpr size_t kHandResidualTemporalConsistencySize =                     //
    kHRTC_HandSize +                                                               //
    kHRTC_RootBoneTranslation +                                                    //
    kHRTC_RootBoneOrientation +                                                    //
    kHRTC_ThumbMCPSwingTwist +                                                     //
    kHRTC_ThumbCurls +                                                             //
    ((kHRTC_FingerMCPSwingTwist + kHRTC_FingerPXMSwing + kHRTC_FingerCurls) * 4) + //
    0;


static constexpr size_t kHandResidualOneSideSize = 21 * 2; //*3;

static constexpr size_t kHandResidualSize = //
    kHandResidualOneSideSize +              //
    kHandResidualOneSideSize +              //
    +kHandResidualTemporalConsistencySize + //
    0;                                      // joints, plus diff hand size from last frame.

// static constexpr size_t kHandResidualSize = 100;

// static constexpr size_t kHandResidualSize = (21 * 3);

template <typename T> struct OptimizerMetacarpalBone
{
	T swing[2]; //[2];
	T twist;
};

template <typename T> struct OptimizerFinger
{
	OptimizerMetacarpalBone<T> metacarpal;
	T proximal_swing[2];
	T rots[2];
};

template <typename T> struct OptimizerThumb
{
	OptimizerMetacarpalBone<T> metacarpal;
	T rots[2];
};

// Not done below. Do this when you come back

template <typename T> struct OptimizerHand
{
	T hand_size;
	T wrist_location[3];
	// This is constant, a ceres::Rotation.h quat,, taken from last frame.
	T wrist_pre_orientation_quat[4];
	// This is optimized - angle-axis rotation vector. Starts at 0, loss goes up the higher it goes because it
	// indicates more of a rotation.
	T wrist_post_orientation_aax[3];
	OptimizerThumb<T> thumb = {};
	OptimizerFinger<T> finger[4] = {};
};


struct minmax
{
	HandScalar min = 0;
	HandScalar max = 0;
};

class FingerLimit
{
public:
	minmax mcp_swing_x = {};
	minmax mcp_swing_y = {};
	minmax mcp_twist = {};

	minmax pxm_swing_x = {};
	minmax pxm_swing_y = {};

	minmax curls[2] = {}; // int, dst
};

class HandLimit
{
public:
	minmax hand_size;

	minmax thumb_mcp_swing_x, thumb_mcp_swing_y, thumb_mcp_twist;
	minmax thumb_curls[2];

	FingerLimit fingers[4];

	HandLimit()
	{
		hand_size = {0.095 - 0.03, 0.095 + 0.03};

		thumb_mcp_swing_x = {rad<HandScalar>(-60), rad<HandScalar>(60)};
		thumb_mcp_swing_y = {rad<HandScalar>(-60), rad<HandScalar>(60)};
		thumb_mcp_twist = {rad<HandScalar>(-90), rad<HandScalar>(90)};

		for (int i = 0; i < 2; i++) {
			thumb_curls[i] = {rad<HandScalar>(-90), rad<HandScalar>(40)};
		}


#define margin 0.09

		fingers[0].mcp_swing_y = {-0.19 - margin, -0.19 + margin};
		fingers[1].mcp_swing_y = {0.00 - margin, 0.00 + margin};
		fingers[2].mcp_swing_y = {0.19 - margin, 0.19 + margin};
		fingers[3].mcp_swing_y = {0.38 - margin, 0.38 + margin};

#undef margin

		for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
			FingerLimit &finger = fingers[finger_idx];

			finger.mcp_swing_x = {rad<HandScalar>(-10), rad<HandScalar>(10)};
			finger.mcp_twist = {rad<HandScalar>(-4), rad<HandScalar>(4)};

			finger.pxm_swing_x = {rad<HandScalar>(-100), rad<HandScalar>(20)}; // ??? why is it reversed
			finger.pxm_swing_y = {rad<HandScalar>(-20), rad<HandScalar>(20)};

			for (int i = 0; i < 2; i++) {
				finger.curls[i] = {rad<HandScalar>(-90), rad<HandScalar>(10)};
			}
		}
	}
};

static const class HandLimit the_limit = {};


constexpr HandScalar hand_size_min = 0.095 - 0.03;
constexpr HandScalar hand_size_max = 0.095 + 0.03;

template <typename T>
inline T
LMToModel(T lm, minmax mm)
{
	return mm.min + ((sin(lm) + T(1)) * ((mm.max - mm.min) * T(.5)));
}

template <typename T>
inline T
ModelToLM(T model, minmax mm)
{
	return asin((2 * (model - mm.min) / (mm.max - mm.min)) - 1);
}

template <typename T>
void
OptimizerHandUnpackFromVector(const T *in, OptimizerHand<T> &out)
{

	size_t acc_idx = 0;

#ifdef USE_HAND_SIZE
	out.hand_size = LMToModel(in[acc_idx++], the_limit.hand_size);
#endif

#ifdef USE_HAND_TRANSLATION
	out.wrist_location[0] = in[acc_idx++];
	out.wrist_location[1] = in[acc_idx++];
	out.wrist_location[2] = in[acc_idx++];
#endif
#ifdef USE_HAND_ORIENTATION
	out.wrist_post_orientation_aax[0] = in[acc_idx++];
	out.wrist_post_orientation_aax[1] = in[acc_idx++];
	out.wrist_post_orientation_aax[2] = in[acc_idx++];
#endif

#ifdef USE_EVERYTHING_ELSE

	out.thumb.metacarpal.swing[0] = LMToModel(in[acc_idx++], the_limit.thumb_mcp_swing_x);
	out.thumb.metacarpal.swing[1] = LMToModel(in[acc_idx++], the_limit.thumb_mcp_swing_y);
	out.thumb.metacarpal.twist = LMToModel(in[acc_idx++], the_limit.thumb_mcp_twist);

	out.thumb.rots[0] = LMToModel(in[acc_idx++], the_limit.thumb_curls[0]);
	out.thumb.rots[1] = LMToModel(in[acc_idx++], the_limit.thumb_curls[1]);

	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {

		out.finger[finger_idx].metacarpal.swing[0] =
		    LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].mcp_swing_x);

		out.finger[finger_idx].metacarpal.swing[1] =
		    LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].mcp_swing_y);

		out.finger[finger_idx].metacarpal.twist =
		    LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].mcp_twist);


		out.finger[finger_idx].proximal_swing[0] =
		    LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].pxm_swing_x);
		out.finger[finger_idx].proximal_swing[1] =
		    LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].pxm_swing_y);

		out.finger[finger_idx].rots[0] = LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].curls[0]);
		out.finger[finger_idx].rots[1] = LMToModel(in[acc_idx++], the_limit.fingers[finger_idx].curls[1]);
	}
#endif
}

template <typename T>
void
OptimizerHandPackIntoVector(OptimizerHand<T> &in, T *out)
{
	size_t acc_idx = 0;
#ifdef USE_HAND_SIZE
	out[acc_idx++] = ModelToLM(in.hand_size, the_limit.hand_size);
#endif

#ifdef USE_HAND_TRANSLATION
	out[acc_idx++] = in.wrist_location[0];
	out[acc_idx++] = in.wrist_location[1];
	out[acc_idx++] = in.wrist_location[2];
#endif
#ifdef USE_HAND_ORIENTATION
	out[acc_idx++] = in.wrist_post_orientation_aax[0];
	out[acc_idx++] = in.wrist_post_orientation_aax[1];
	out[acc_idx++] = in.wrist_post_orientation_aax[2];
#endif
#ifdef USE_EVERYTHING_ELSE
	out[acc_idx++] = ModelToLM(in.thumb.metacarpal.swing[0], the_limit.thumb_mcp_swing_x);
	out[acc_idx++] = ModelToLM(in.thumb.metacarpal.swing[1], the_limit.thumb_mcp_swing_y);
	out[acc_idx++] = ModelToLM(in.thumb.metacarpal.twist, the_limit.thumb_mcp_twist);

	out[acc_idx++] = ModelToLM(in.thumb.rots[0], the_limit.thumb_curls[0]);
	out[acc_idx++] = ModelToLM(in.thumb.rots[1], the_limit.thumb_curls[1]);

	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		out[acc_idx++] =
		    ModelToLM(in.finger[finger_idx].metacarpal.swing[0], the_limit.fingers[finger_idx].mcp_swing_x);
		out[acc_idx++] =
		    ModelToLM(in.finger[finger_idx].metacarpal.swing[1], the_limit.fingers[finger_idx].mcp_swing_y);
		out[acc_idx++] =
		    ModelToLM(in.finger[finger_idx].metacarpal.twist, the_limit.fingers[finger_idx].mcp_twist);

		out[acc_idx++] =
		    ModelToLM(in.finger[finger_idx].proximal_swing[0], the_limit.fingers[finger_idx].pxm_swing_x);
		out[acc_idx++] =
		    ModelToLM(in.finger[finger_idx].proximal_swing[1], the_limit.fingers[finger_idx].pxm_swing_y);

		out[acc_idx++] = ModelToLM(in.finger[finger_idx].rots[0], the_limit.fingers[finger_idx].curls[0]);
		out[acc_idx++] = ModelToLM(in.finger[finger_idx].rots[1], the_limit.fingers[finger_idx].curls[1]);
	}
#endif
}

template <typename T>
void
OptimizerHandInit(OptimizerHand<T> &opt, HandScalar *pre_rotation)
{
	opt.hand_size = (T)(0.095);

	opt.wrist_post_orientation_aax[0] = (T)(0);
	opt.wrist_post_orientation_aax[1] = (T)(0);
	opt.wrist_post_orientation_aax[2] = (T)(0);

	// opt.store_wrist_pre_orientation_quat = pre_rotation;

	opt.wrist_pre_orientation_quat[0] = (T)pre_rotation[0];
	opt.wrist_pre_orientation_quat[1] = (T)pre_rotation[1];
	opt.wrist_pre_orientation_quat[2] = (T)pre_rotation[2];
	opt.wrist_pre_orientation_quat[3] = (T)pre_rotation[3];

	opt.wrist_location[0] = (T)(0);
	opt.wrist_location[1] = (T)(0);
	opt.wrist_location[2] = (T)(-0.3);


	for (int i = 0; i < 4; i++) {
		opt.finger[i].proximal_swing[0] = rad<T>((T)(15));
		opt.finger[i].rots[0] = rad<T>((T)(-5));
		opt.finger[i].rots[1] = rad<T>((T)(-5));
	}

	opt.thumb.metacarpal.swing[0] = (T)(0);
	opt.thumb.metacarpal.swing[1] = (T)(0);
	opt.thumb.metacarpal.twist = (T)(0);

	opt.thumb.rots[0] = rad<T>((T)(-5));
	opt.thumb.rots[1] = rad<T>((T)(-59));

	opt.finger[0].metacarpal.swing[1] = (T)(-0.19);
	opt.finger[1].metacarpal.swing[1] = (T)(0);
	opt.finger[2].metacarpal.swing[1] = (T)(0.19);
	opt.finger[3].metacarpal.swing[1] = (T)(0.38);

	opt.finger[0].proximal_swing[1] = (T)(-0.01);
	opt.finger[1].proximal_swing[1] = (T)(0);
	opt.finger[2].proximal_swing[1] = (T)(0.01);
	opt.finger[3].proximal_swing[1] = (T)(0.02);
}



// Applies the post axis-angle rotation to the pre quat, then zeroes out the axis-angle.
template <typename T>
void
OptimizerHandSquashRotations(OptimizerHand<T> &opt, HandScalar *out_orientation)
{

	// Hmmmmm, is this at all the right thing to do? :
	opt.wrist_pre_orientation_quat[0] = (T)out_orientation[0];
	opt.wrist_pre_orientation_quat[1] = (T)out_orientation[1];
	opt.wrist_pre_orientation_quat[2] = (T)out_orientation[2];
	opt.wrist_pre_orientation_quat[3] = (T)out_orientation[3];

	T *pre_rotation = opt.wrist_pre_orientation_quat;

	T post_rotation[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax, post_rotation);

	T tmp_new_pre_rotation[4];

	ceres::QuaternionProduct(pre_rotation, post_rotation, tmp_new_pre_rotation);

	for (int i = 0; i < 4; i++) {
		out_orientation[i] = tmp_new_pre_rotation[i];
	}


	for (int i = 0; i < 3; i++) {
		opt.wrist_post_orientation_aax[i] = (T)(0);
	}
}
