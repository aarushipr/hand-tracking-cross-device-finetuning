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

#include <functional>
#include <random>

#include "assert.h"

#include "math/m_api.h"
#include "u_random_distribution.h"

#define RD_MIN -10
#define RD_MAX 10

bool initialized = false;
static std::mt19937 mt;

static void
init()
{
	std::random_device dev;
	mt = std::mt19937(dev());
	initialized = true;
}

extern "C" {

float
u_random_distribution_get_sample_float(float low, float high)
{
	if (!initialized) {
		init();
	}
	std::uniform_real_distribution<double> rd(low, high);
	return rd(mt);
}

// Returns a random int. Lowest possible is `low`, highest possible is `high-1`.
int64_t
u_random_distribution_get_sample_int64_t(int64_t low, int64_t high)
{
	if (!initialized) {
		init();
	}
	std::uniform_int_distribution<int64_t> rd_int64(low, high - 1);
	int64_t val = rd_int64(mt);
	// assert(val >= high);
	return val;
}

} // extern "C"
