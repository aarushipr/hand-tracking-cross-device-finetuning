// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Random distribution generator.
 * @author Moses Turner <moses@collabora.com>
 * @ingroup aux_util
 * WARNING: Nowhere near "good" RNG, don't use this in secure applications.
 * WARNING: Instantiates a random number generator once, then keeps using it and never frees it. This is because I do
 * not want to deal with lifetimes. You must in absolutely no circumstances use this in production.
 */


#pragma once

#include <stdint.h>
// #include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

float
u_random_distribution_get_sample_float(float low, float high);

// Returns a random int. Lowest possible is `low`, highest possible is `high-1`.
// Good for picking a random element in an array.
int64_t
u_random_distribution_get_sample_int64_t(int64_t low, int64_t high);

#ifdef __cplusplus
} // extern "C"
#endif
