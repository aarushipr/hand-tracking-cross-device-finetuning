#include "util/u_logging.h"
#include "math/m_api.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

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
#ifdef USE_HAND_SIZE
    kHandSizeDim + //
#endif
#ifdef USE_HAND_TRANSLATION
    kHandTranslationDim + //
#endif
#ifdef USE_HAND_ORIENTATION
    kHandOrientationDim + //
#endif
#ifdef USE_EVERYTHING_ELSE
    kThumbDim +      //
    (kFingerDim * 4) //
#endif
    ;

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
static constexpr size_t kOutputErrorVectorSize =
    (21 * 3) + 1 + 3 + 2 + ((3 + 2 + 2) * 4) + 1 + 3; // joints, plus diff hand size from last frame.

// static constexpr size_t kOutputErrorVectorSize = 21*3;

template <typename HandScalar> class OptimizerMetacarpalBone
{
private:
	HandScalar *swing_ptr; //[2];
	HandScalar *twist_ptr;

public:
	// OptimizerMetacarpalBone(Eigen::Map<Eigen::Vector<HandScalar, 3>> in) : m_block(&in(0)) {}
	inline HandScalar *
	swing()
	{
		return swing_ptr;
	}

	inline HandScalar *
	twist()
	{
		return twist_ptr;
	}
	// template <typename HandScalar>
	OptimizerMetacarpalBone(HandScalar *root)
	{
		swing_ptr = root;
		twist_ptr = &root[2];
	}
};

template <typename HandScalar> class OptimizerFinger
{
private:
	OptimizerMetacarpalBone<HandScalar> store_mcp;
	HandScalar pxm_swing_ptr[2];
	HandScalar rots_ptr[2];

public:
	OptimizerMetacarpalBone<HandScalar> &
	metacarpal()
	{
		return store_mcp;
	}
	inline HandScalar *
	proximal_swing()
	{
		return pxm_swing_ptr;
	}

	inline HandScalar *
	rots()
	{
		return rots_ptr;
	}

	OptimizerFinger(HandScalar *root) : store_mcp(&root[0])
	{
		pxm_swing_ptr = &root[kMetacarpalBoneDim];
		rots_ptr = &root[kMetacarpalBoneDim + kProximalBoneDim];
	}
};

template <typename HandScalar> class OptimizerThumb
{
	OptimizerMetacarpalBone<HandScalar> store_mcp;
	HandScalar rots_ptr[2];

public:
	OptimizerMetacarpalBone<HandScalar> &
	metacarpal()
	{
		return store_mcp;
	}
	inline HandScalar *
	rots()
	{
		return rots_ptr;
	}
};

// Not done below. Do this when you come back

template <typename HandScalar> class OptimizerHand
{
private:
	HandScalar *hand_size_ptr;
	HandScalar *wrist_location_ptr;
	HandScalar *wrist_orientation_axisangle;
	OptimizerThumb<HandScalar> store_thumb = {};
	OptimizerFinger<HandScalar> store_finger[4] = {};

public:
	// OptimizerHand(Eigen::Vector<HandScalar, kHandDim> &in) : m_block(&in(0)) {}
	// OptimizerHand(Eigen::Vector<HandScalar, Eigen::Dynamic> &in) : m_block(&in(0)) {}

	bool was_initialized = false;

	inline HandScalar *
	hand_size()
	{
		return hand_size_ptr;
	};

	inline HandScalar *
	wrist_location()
	{
		return wrist_location_ptr;
	}

	inline Eigen::Vector3<HandScalar> &
	wrist_orientation_rodrigues()
	{
		return store_wrist_orientation_rodrigues;
	}

	inline OptimizerThumb &
	thumb()
	{
		return store_thumb;
	}

	template <int idx>
	OptimizerFinger &
	finger()
	{
		return store_finger[idx];
	}
	OptimizerFinger &
	finger(int idx)
	{
		return store_finger[idx];
	}
	Optimizer
};


class minmax
{
public:
	double min = 0;
	double max = 0;
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

		thumb_mcp_swing_x = {rad(-60), rad(60)};
		thumb_mcp_swing_y = {rad(-60), rad(60)};
		thumb_mcp_twist = {rad(-90), rad(90)};

		for (int i = 0; i < 2; i++) {
			thumb_curls[i] = {rad(-90), rad(40)};
		}


#define margin 0.09

		fingers[0].mcp_swing_y = {-0.19 - margin, -0.19 + margin};
		fingers[1].mcp_swing_y = {0.00 - margin, 0.00 + margin};
		fingers[2].mcp_swing_y = {0.19 - margin, 0.19 + margin};
		fingers[3].mcp_swing_y = {0.38 - margin, 0.38 + margin};

#undef margin

		for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
			FingerLimit &finger = fingers[finger_idx];

			finger.mcp_swing_x = {rad(-10), rad(10)};
			finger.mcp_twist = {rad(-4), rad(4)};

			finger.pxm_swing_x = {rad(-100), rad(20)}; // ??? why is it reversed
			finger.pxm_swing_y = {rad(-20), rad(20)};

			for (int i = 0; i < 2; i++) {
				finger.curls[i] = {rad(-90), rad(10)};
			}
		}
	}
};

static const class HandLimit the_limit = {};


constexpr double hand_size_min = 0.095 - 0.03;
constexpr double hand_size_max = 0.095 + 0.03;

double
LMToModel(double lm, minmax mm)
{
	return mm.min + ((std::sin(lm) + 1) * ((mm.max - mm.min) * .5));
}

double
ModelToLM(double model, minmax mm)
{
	return std::asin((2 * (model - mm.min) / (mm.max - mm.min)) - 1);
}


void
SmartUnpackFromVector(const Eigen::VectorXd &in, OptimizerHand &out)
{

	size_t acc_idx = 0;

	// ?????
	out.hand_size() = LMToModel(in(acc_idx++), the_limit.hand_size);


	out.wrist_location().x() = in(acc_idx++);
	out.wrist_location().y() = in(acc_idx++);
	out.wrist_location().z() = in(acc_idx++);

	out.wrist_orientation_rodrigues().x() = in(acc_idx++);
	out.wrist_orientation_rodrigues().y() = in(acc_idx++);
	out.wrist_orientation_rodrigues().z() = in(acc_idx++);

	out.thumb().metacarpal().swing().x() = LMToModel(in(acc_idx++), the_limit.thumb_mcp_swing_x);
	out.thumb().metacarpal().swing().y() = LMToModel(in(acc_idx++), the_limit.thumb_mcp_swing_y);
	out.thumb().metacarpal().twist() = LMToModel(in(acc_idx++), the_limit.thumb_mcp_twist);

	out.thumb().rots()(0) = LMToModel(in(acc_idx++), the_limit.thumb_curls[0]);
	out.thumb().rots()(1) = LMToModel(in(acc_idx++), the_limit.thumb_curls[1]);

	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {

		out.finger(finger_idx).metacarpal().swing().x() =
		    LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].mcp_swing_x);

		out.finger(finger_idx).metacarpal().swing().y() =
		    LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].mcp_swing_y);

		out.finger(finger_idx).metacarpal().twist() =
		    LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].mcp_twist);


		out.finger(finger_idx).proximal_swing().x() =
		    LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].pxm_swing_x);
		out.finger(finger_idx).proximal_swing().y() =
		    LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].pxm_swing_y);

		out.finger(finger_idx).rots()(0) = LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].curls[0]);
		out.finger(finger_idx).rots()(1) = LMToModel(in(acc_idx++), the_limit.fingers[finger_idx].curls[1]);
	}
}

void
SmartPackIntoVector(OptimizerHand &in, Eigen::VectorXd &out)
{
	size_t acc_idx = 0;

	out(acc_idx++) = ModelToLM(in.hand_size(), the_limit.hand_size);

	out(acc_idx++) = in.wrist_location().x();
	out(acc_idx++) = in.wrist_location().y();
	out(acc_idx++) = in.wrist_location().z();

	out(acc_idx++) = in.wrist_orientation_rodrigues().x();
	out(acc_idx++) = in.wrist_orientation_rodrigues().y();
	out(acc_idx++) = in.wrist_orientation_rodrigues().z();

	out(acc_idx++) = ModelToLM(in.thumb().metacarpal().swing().x(), the_limit.thumb_mcp_swing_x);
	out(acc_idx++) = ModelToLM(in.thumb().metacarpal().swing().y(), the_limit.thumb_mcp_swing_y);
	out(acc_idx++) = ModelToLM(in.thumb().metacarpal().twist(), the_limit.thumb_mcp_twist);

	out(acc_idx++) = ModelToLM(in.thumb().rots()(0), the_limit.thumb_curls[0]);
	out(acc_idx++) = ModelToLM(in.thumb().rots()(1), the_limit.thumb_curls[1]);

	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		out(acc_idx++) = ModelToLM(in.finger(finger_idx).metacarpal().swing().x(),
		                           the_limit.fingers[finger_idx].mcp_swing_x);
		out(acc_idx++) = ModelToLM(in.finger(finger_idx).metacarpal().swing().y(),
		                           the_limit.fingers[finger_idx].mcp_swing_y);
		out(acc_idx++) =
		    ModelToLM(in.finger(finger_idx).metacarpal().twist(), the_limit.fingers[finger_idx].mcp_twist);

		out(acc_idx++) =
		    ModelToLM(in.finger(finger_idx).proximal_swing().x(), the_limit.fingers[finger_idx].pxm_swing_x);
		out(acc_idx++) =
		    ModelToLM(in.finger(finger_idx).proximal_swing().y(), the_limit.fingers[finger_idx].pxm_swing_y);

		out(acc_idx++) = ModelToLM(in.finger(finger_idx).rots()(0), the_limit.fingers[finger_idx].curls[0]);
		out(acc_idx++) = ModelToLM(in.finger(finger_idx).rots()(1), the_limit.fingers[finger_idx].curls[1]);
	}
}