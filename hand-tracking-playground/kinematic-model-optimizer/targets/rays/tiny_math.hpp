// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Math for kinematic model
 * @author Moses Turner <moses@collabora.com>
 * @ingroup tracking
 */

#pragma once

#include "math/m_api.h"
#include "math/m_mathinclude.h"
#include "defines.hpp"

// // Waggle-curl-twist.
// static inline void
// wct_to_quat(wct_t wct, struct xrt_quat *out)
// {
// 	xrt_vec3 waggle_axis = {0, 1, 0};
// 	xrt_quat just_waggle;
// 	math_quat_from_angle_vector(wct.waggle, &waggle_axis, &just_waggle);

// 	xrt_vec3 curl_axis = {1, 0, 0};
// 	xrt_quat just_curl;
// 	math_quat_from_angle_vector(wct.curl, &curl_axis, &just_curl);

// 	xrt_vec3 twist_axis = {0, 0, 1};
// 	xrt_quat just_twist;
// 	math_quat_from_angle_vector(wct.twist, &twist_axis, &just_twist);

// 	//! @todo: optimize This should be a matrix multiplication...
// 	*out = just_waggle;
// 	math_quat_rotate(out, &just_curl, out);
// 	math_quat_rotate(out, &just_twist, out);
// }

// Inlines.
template <typename T>
T
rad(T degrees)
{
	return degrees * T(M_PI / 180.0);
}


template <typename S>
void
clamp(S &in, S min, S max)
{
	in = std::min(max, std::max(min, in));
}

template <typename S>
void
clamp_to_r(S &in, S c, S r)
{
	clamp<S>(in, c - r, c + r);
}
