#!/bin/bash
# Copyright 2023, Collabora, Ltd.
# SPDX-License-Identifier: MIT or BSL-1.0 or Apache-2.0

cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX="/usr"

ninja -C build package

dpkg -c build/*.deb
