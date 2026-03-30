#!/bin/sh
# Copyright 2019-2021, Collabora, Ltd.
# SPDX-License-Identifier: BSL-1.0
# Author: Ryan Pavlik <ryan.pavlik@collabora.com>

# Formats all the source files in this project

set -e

AUTOPEP8=autopep8

(
        ${AUTOPEP8} --version

        cd $(dirname $0)/..

        find \
                py/ \
                \( -name "*.py" \) \
                -exec ${AUTOPEP8} --in-place --aggressive --aggressive \{\} +
)
