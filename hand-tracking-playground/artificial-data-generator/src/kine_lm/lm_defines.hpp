// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Defines for Levenberg-Marquardt kinematic optimizer
 * @author Moses Turner <moses@collabora.com>
 * @ingroup tracking
 */
#pragma once

// #include <Eigen/Core>
// #include <Eigen/Geometry>
#include "util/u_logging.h"
#include "math/m_mathinclude.h"
#include "../kine_common.hpp"
#include <type_traits>

namespace xrt::tracking::hand::mercury::lm {

#define LM_TRACE(lmh, ...) U_LOG_IFL_T(lmh.log_level, __VA_ARGS__)
#define LM_DEBUG(lmh, ...) U_LOG_IFL_D(lmh.log_level, __VA_ARGS__)
#define LM_INFO(lmh, ...) U_LOG_IFL_I(lmh.log_level, __VA_ARGS__)
#define LM_WARN(lmh, ...) U_LOG_IFL_W(lmh.log_level, __VA_ARGS__)
#define LM_ERROR(lmh, ...) U_LOG_IFL_E(lmh.log_level, __VA_ARGS__)

// Inlines.
template <typename T>
inline T
rad(T degrees)
{
	return degrees * T(M_PI / 180.f);
}

// Number of joints that our ML models output.
static constexpr size_t kNumNNJoints = 21;

static constexpr size_t kNumFingers = 5;

// This is a lie for the thumb; we usually do the hidden metacarpal trick there
static constexpr size_t kNumJointsInFinger = 5;

static constexpr size_t kNumOrientationsInFinger = 5;

// These defines look silly, but they are _extremely_ useful for doing work on this optimizer. Please don't remove them.
#define USE_HAND_SIZE
#undef USE_HAND_TRANSLATION
#undef USE_HAND_ORIENTATION
#define USE_EVERYTHING_ELSE

// Not tested/tuned well enough; might make tracking slow.
#define USE_HAND_PLAUSIBILITY

static constexpr size_t kMetacarpalBoneDim = 3;
static constexpr size_t kProximalBoneDim = 2;
static constexpr size_t kFingerDim = kProximalBoneDim + kMetacarpalBoneDim + 2;
static constexpr size_t kThumbDim = kMetacarpalBoneDim + 2;
static constexpr size_t kHandSizeDim = 1;
static constexpr size_t kHandTranslationDim = 3;
static constexpr size_t kHandOrientationDim = 3;



static constexpr size_t kHRTC_HandSize = 1;
static constexpr size_t kHRTC_RootBoneTranslation = 3;
static constexpr size_t kHRTC_RootBoneOrientation = 3; // Direct difference between the two angle-axis rotations. This
                                                       // works well enough because the rotation should be small.

static constexpr size_t kHRTC_ThumbMCPSwingTwist = 3;
static constexpr size_t kHRTC_ThumbCurls = 2;

static constexpr size_t kHRTC_ProximalSimilarity = 2;

static constexpr size_t kHRTC_FingerMCPSwingTwist = 3;
static constexpr size_t kHRTC_FingerPXMSwing = 2;
static constexpr size_t kHRTC_FingerCurls = 2;
static constexpr size_t kHRTC_CurlSimilarity = 1;

static constexpr size_t kHandResidualOneSideSize = 21 * 2;

static constexpr size_t kHandResidualTemporalConsistencyOneFingerSize = //
    kHRTC_FingerMCPSwingTwist +                                         //
    kHRTC_FingerPXMSwing +                                              //
    kHRTC_FingerCurls +                                                 //
#ifdef USE_HAND_PLAUSIBILITY                                            //
    kHRTC_CurlSimilarity +                                              //
#endif                                                                  //
    0;

static constexpr size_t kHandResidualTemporalConsistencySize = //
    kHRTC_RootBoneTranslation +                                //
    kHRTC_RootBoneOrientation +                                //
    kHRTC_ThumbMCPSwingTwist +                                 //
    kHRTC_ThumbCurls +                                         //
#ifdef USE_HAND_PLAUSIBILITY                                   //
    kHRTC_ProximalSimilarity +                                 //
#endif                                                         //
    (kHandResidualTemporalConsistencyOneFingerSize * 4) +      //
    0;


// Factors to multiply different values by to get a smooth hand trajectory without introducing too much latency

// 1.0 is good, a little jittery.
// Anything above 3.0 generally breaks.
static constexpr HandScalar kStabilityRoot = 80.0;
static constexpr HandScalar kStabilityCurlRoot = kStabilityRoot * 0.03f;
static constexpr HandScalar kStabilityOtherRoot = kStabilityRoot * 0.03f;

static constexpr HandScalar kStabilityThumbMCPSwing = kStabilityCurlRoot * 1.5f;
static constexpr HandScalar kStabilityThumbMCPTwist = kStabilityCurlRoot * 1.5f;

static constexpr HandScalar kStabilityFingerMCPSwing = kStabilityCurlRoot * 3.0f;
static constexpr HandScalar kStabilityFingerMCPTwist = kStabilityCurlRoot * 3.0f;

static constexpr HandScalar kStabilityFingerPXMSwingX = kStabilityCurlRoot * 1.0f;
static constexpr HandScalar kStabilityFingerPXMSwingY = kStabilityCurlRoot * 1.6f;

static constexpr HandScalar kStabilityRootPosition = kStabilityOtherRoot * 30;
static constexpr HandScalar kStabilityHandSize = kStabilityOtherRoot * 1000;

static constexpr HandScalar kStabilityHandOrientation = kStabilityOtherRoot * 3;


static constexpr HandScalar kPlausibilityRoot = 0.3;
static constexpr HandScalar kPlausibilityCurlSimilarity_IndexMiddle = 0.3f * kPlausibilityRoot;
static constexpr HandScalar kPlausibilityCurlSimilarity_MiddleRing = 2.0f * kPlausibilityRoot;
static constexpr HandScalar kPlausibilityCurlSimilarity_RingLittle = 2.5f * kPlausibilityRoot;

static constexpr HandScalar kPlausibilityCurlSimilarityHard = 0.10f * kPlausibilityRoot;
static constexpr HandScalar kPlausibilityCurlSimilaritySoft = 0.05f * kPlausibilityRoot;


constexpr size_t
calc_input_size(bool optimize_hand_size)
{
	size_t out = 0;

#ifdef USE_HAND_TRANSLATION
	out += kHandTranslationDim;
#endif

#ifdef USE_HAND_ORIENTATION
	out += kHandOrientationDim;
#endif

#ifdef USE_EVERYTHING_ELSE
	out += kThumbDim;
	out += (kFingerDim * 4);
#endif

#ifdef USE_HAND_SIZE
	if (optimize_hand_size) {
		out += kHandSizeDim;
	}
#endif

	return out;
}


constexpr size_t
calc_residual_size(bool stability)
{
	size_t out = 0;

	out += 500;
	if (stability) {
		out += kHandResidualTemporalConsistencySize;
	}
	return out;
}

// Some templatable spatial types.
// Heavily inspired by Eigen - one can definitely use Eigen instead, but here I'd rather have more control

template <typename Scalar> struct Quat
{
	Scalar x;
	Scalar y;
	Scalar z;
	Scalar w;

	/// Default constructor - DOES NOT INITIALIZE VALUES
	constexpr Quat() {}

	/// Copy constructor
	constexpr Quat(Quat const &) noexcept(std::is_nothrow_copy_constructible_v<Scalar>) = default;

	/// Move constructor
	Quat(Quat &&) noexcept(std::is_nothrow_move_constructible_v<Scalar>) = default;

	/// Copy assignment
	Quat &
	operator=(Quat const &) = default;

	/// Move assignment
	Quat &
	operator=(Quat &&) noexcept = default;

	/// Construct from x, y, z, w scalars
	template <typename Other>
	constexpr Quat(Other x, Other y, Other z, Other w) noexcept // NOLINT(bugprone-easily-swappable-parameters)
	    : x{Scalar(x)}, y{Scalar(y)}, z{Scalar(z)}, w{Scalar(w)}
	{}

	/// So that we can copy a regular Vec2 into the real part of a Jet Vec2
	template <typename Other> Quat(Quat<Other> const &other) : Quat(other.x, other.y, other.z, other.w) {}
	
	Quat(xrt_quat const &other) : Quat(other.x, other.y, other.z, other.w) {}

	static Quat
	Identity()
	{
		return Quat(0.f, 0.f, 0.f, 1.f);
	}
};

template <typename Scalar> struct Vec3
{
	// Note that these are not initialized, for performance reasons.
	// If you want them initialized, use Zero() or something else
	Scalar x;
	Scalar y;
	Scalar z;

	/// Default constructor - DOES NOT INITIALIZE VALUES
	constexpr Vec3() {}
	/// Copy constructor
	constexpr Vec3(Vec3 const &other) noexcept(std::is_nothrow_copy_constructible_v<Scalar>) = default;

	/// Move constructor
	Vec3(Vec3 &&) noexcept(std::is_nothrow_move_constructible_v<Scalar>) = default;

	/// Copy assignment
	Vec3 &
	operator=(Vec3 const &) = default;

	/// Move assignment
	Vec3 &
	operator=(Vec3 &&) noexcept = default;


	template <typename Other>
	constexpr Vec3(Other x, Other y, Other z) noexcept // NOLINT(bugprone-easily-swappable-parameters)
	    : x{Scalar(x)}, y{Scalar(y)}, z{Scalar(z)}
	{}

	template <typename Other> Vec3(Vec3<Other> const &other) : Vec3(other.x, other.y, other.z) {}

	static Vec3
	Zero()
	{
		return Vec3(0.f, 0.f, 0.f);
	}
};

template <typename Scalar> struct Vec2
{
	Scalar x;
	Scalar y;

	/// Default constructor - DOES NOT INITIALIZE VALUES
	constexpr Vec2() noexcept {}

	/// Copy constructor
	constexpr Vec2(Vec2 const &) noexcept(std::is_nothrow_copy_constructible_v<Scalar>) = default;

	/// Move constructor
	constexpr Vec2(Vec2 &&) noexcept(std::is_nothrow_move_constructible_v<Scalar>) = default;

	/// Copy assignment
	Vec2 &
	operator=(Vec2 const &) = default;

	/// Move assignment
	Vec2 &
	operator=(Vec2 &&) noexcept = default;

	/// So that we can copy a regular Vec2 into the real part of a Jet Vec2
	template <typename Other>
	Vec2(Other x, Other y) // NOLINT(bugprone-easily-swappable-parameters)
	    noexcept(std::is_nothrow_constructible_v<Scalar, Other>)
	    : x{Scalar(x)}, y{Scalar(y)}
	{}

	template <typename Other>
	Vec2(Vec2<Other> const &other) noexcept(std::is_nothrow_constructible_v<Scalar, Other>) : Vec2(other.x, other.y)
	{}

	static constexpr Vec2
	Zero()
	{
		return Vec2(0.f, 0.f);
	}
};

template <typename T> struct ResidualHelper
{
	T *out_residual;
	size_t out_residual_idx = 0;
	size_t max_size = 0;

	ResidualHelper(T *residual, size_t max_size) : out_residual(residual), max_size(max_size)
	{
		out_residual_idx = 0;
	}

	void
	AddValue(T const &value)
	{
		if (out_residual_idx == this->max_size) {
			U_LOG_E("NO! has size %zu max size %zu", out_residual_idx, this->max_size);
			abort();
		}
		// U_LOG_E("CRY...");
		this->out_residual[out_residual_idx++] = value;
	}
};

static enum xrt_hand_joint joints_5x5_to_26[5][5] = {
    {
        XRT_HAND_JOINT_WRIST,
        XRT_HAND_JOINT_THUMB_METACARPAL,
        XRT_HAND_JOINT_THUMB_PROXIMAL,
        XRT_HAND_JOINT_THUMB_DISTAL,
        XRT_HAND_JOINT_THUMB_TIP,
    },
    {
        XRT_HAND_JOINT_INDEX_METACARPAL,
        XRT_HAND_JOINT_INDEX_PROXIMAL,
        XRT_HAND_JOINT_INDEX_INTERMEDIATE,
        XRT_HAND_JOINT_INDEX_DISTAL,
        XRT_HAND_JOINT_INDEX_TIP,
    },
    {
        XRT_HAND_JOINT_MIDDLE_METACARPAL,
        XRT_HAND_JOINT_MIDDLE_PROXIMAL,
        XRT_HAND_JOINT_MIDDLE_INTERMEDIATE,
        XRT_HAND_JOINT_MIDDLE_DISTAL,
        XRT_HAND_JOINT_MIDDLE_TIP,
    },
    {
        XRT_HAND_JOINT_RING_METACARPAL,
        XRT_HAND_JOINT_RING_PROXIMAL,
        XRT_HAND_JOINT_RING_INTERMEDIATE,
        XRT_HAND_JOINT_RING_DISTAL,
        XRT_HAND_JOINT_RING_TIP,
    },
    {
        XRT_HAND_JOINT_LITTLE_METACARPAL,
        XRT_HAND_JOINT_LITTLE_PROXIMAL,
        XRT_HAND_JOINT_LITTLE_INTERMEDIATE,
        XRT_HAND_JOINT_LITTLE_DISTAL,
        XRT_HAND_JOINT_LITTLE_TIP,
    }};


template <typename T>
T
vector_length(Vec3<T> vec)
{
	T l_sqrd = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	if (l_sqrd == 0) {
		// at 0, we get nan in the derivative of sqrt
		// rethink?: sqrt's derivative gets reall high close to 0 and it's not great. is there a more stable
		// approximate vector length?
		//
		return vec.x + vec.y + vec.z; // :|
	}
	return sqrt(l_sqrd);
}

// yeah, it segfaults here. but whah?
template <typename T>
Vec3<T>
vector_sub(Vec3<T> vec1, Vec3<T> vec2)
{
	Vec3<T> ret;
	ret.x = vec1.x - vec2.x;
	ret.y = vec1.y - vec2.y;
	ret.z = vec1.z - vec2.z;
	return ret;
}


template <typename T>
T
vector_diff(const Vec3<T> &vec1, const Vec3<T> &vec2)
{
	Vec3<T> diff = vector_sub(vec1, vec2);
	return vector_length(diff);
}

template <typename T>
void
vector_cross(const Vec3<T> &first, const Vec3<T> &second, Vec3<T> &ret)
{
	ret.x = first.y * second.z - first.z * second.y;
	ret.y = -(first.x * second.z - first.z * second.x);
	ret.z = first.x * second.y - first.y * second.x;
}

template <typename T>
T
vector_dot(Vec3<T> first, Vec3<T> second)
{
	return ((first.x * second.x) + (first.y * second.y) + (first.z * second.z));
}

template <typename T> struct Translations55
{
	Vec3<T> t[kNumFingers][kNumJointsInFinger];
};

template <typename T> struct Orientations54
{
	Quat<T> q[kNumFingers][kNumJointsInFinger];
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
		thumb_mcp_twist = {rad<HandScalar>(-35), rad<HandScalar>(35)};

		for (int i = 0; i < 2; i++) {
			thumb_curls[i] = {rad<HandScalar>(-90), rad<HandScalar>(40)};
		}


		constexpr double margin = 0.19;

		for (int i = 0; i < 4; i++) {
			fingers[i].mcp_swing_y = {-margin, margin};
		}

		// fingers[0].mcp_swing_y = {-0.19 - margin, -0.19 + margin};
		// fingers[1].mcp_swing_y = {0.00 - margin, 0.00 + margin};
		// fingers[2].mcp_swing_y = {0.19 - margin, 0.19 + margin};
		// fingers[3].mcp_swing_y = {0.38 - margin, 0.38 + margin};


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

		// Little-proximal was being allowed to swing out too far by my eye
		fingers[3].pxm_swing_y.max = rad<HandScalar>(12);

	}
};


} // namespace xrt::tracking::hand::mercury::lm
