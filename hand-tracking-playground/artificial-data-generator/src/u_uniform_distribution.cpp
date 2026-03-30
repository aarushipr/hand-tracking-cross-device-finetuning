// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Uniform distribution generator.
 * @author Moses Turner <moses@collabora.com>
 * @ingroup aux_util
 * WARNING: Nowhere near "good" RNG, don't use this in secure applications.
 */

#include <functional>
#include <random>

#include "assert.h"

#include "math/m_api.h"
#include "u_uniform_distribution.h"

#define RD_MIN -10
#define RD_MAX 10

struct uniform_distribution
{
	std::mt19937 mt;
};

extern "C" {


void
u_random_distribution_create(struct uniform_distribution **out_distribution)
{
	uniform_distribution *dist_ptr = new uniform_distribution;
	uniform_distribution &dist = *dist_ptr;
	std::random_device dev;

	dist.mt = std::mt19937(dev());
	*out_distribution = dist_ptr;
}

float
u_random_distribution_get_sample_float(struct uniform_distribution *distribution, float low, float high)
{
	// int e = ();
	std::uniform_real_distribution<double> rd(low, high);
	return rd(distribution->mt);

}

// Returns a random int. Lowest possible is `low`, highest possible is `high-1`.
int64_t
u_random_distribution_get_sample_int64_t(struct uniform_distribution *distribution, int64_t low, int64_t high)
{
	std::uniform_int_distribution<int64_t> rd_int64(low, high - 1);
	int64_t val = rd_int64(distribution->mt);
	// assert(val >= high);
	return val;
}

void
u_random_distribution_destroy(struct uniform_distribution **ptr_to_distribution)
{
	delete *ptr_to_distribution;
	ptr_to_distribution = NULL;
}


} // extern "C"
