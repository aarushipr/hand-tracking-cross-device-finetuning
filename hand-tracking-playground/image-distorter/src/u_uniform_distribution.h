// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Uniform distribution generator.
 * @author Moses Turner <moses@collabora.com>
 * @ingroup aux_util
 * WARNING: Nowhere near "good" RNG, don't use this in secure applications.
 */


#pragma once

// #include <stdint.h>
// #include <stddef.h>

struct uniform_distribution;


#ifdef __cplusplus
extern "C" {
#endif

void
u_random_distribution_create(struct uniform_distribution **out_distribution);

float
u_random_distribution_get_sample_float(struct uniform_distribution *distribution, float low, float high);

// Returns a random int. Lowest possible is `low`, highest possible is `high-1`.
// Good for picking a random element in an array.
int
u_random_distribution_get_sample_int(struct uniform_distribution *distribution, int low, int high);

void
u_random_distribution_destroy(struct uniform_distribution **ptr_to_distribution);

#ifdef __cplusplus
} // extern "C"
#endif
