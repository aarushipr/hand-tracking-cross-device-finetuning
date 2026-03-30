#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef double HandScalar;
typedef Eigen::Transform<HandScalar, 3, Eigen::Affine> PODTrans;


enum HandFinger
{
	HF_THUMB = 0,
	HF_INDEX = 1,
	HF_MIDDLE = 2,
	HF_RING = 3,
	HF_LITTLE = 4,
};

enum FingerBone
{
	FB_METACARPAL,
	FB_PROXIMAL,
	FB_INTERMEDIATE,
	FB_DISTAL
};

enum ThumbBone
{
	TB_METACARPAL,
	TB_PROXIMAL,
	TB_DISTAL
};


constexpr double default_hand_size_m = 0.095;