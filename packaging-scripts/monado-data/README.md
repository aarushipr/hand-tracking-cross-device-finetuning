# Monado Data

<!--
Copyright 2023, Collabora, Ltd.
SPDX-License-Identifier: MIT or BSL-1.0 or Apache-2.0
-->

A hack that lets you produce debian packages for the Monado ML models and other
data static data needed for Monado (currently only hand tracking models). This
uses CMake's CPack. Yes really CMake's CPack, it's silly and dumb but hey it
works and was fairly easy to do. Could probably extend to other distros, so
patches very much welcome. Run `./make-package.sh` to create the package.

#### Dependancies

"Compile" time dependancies
```bash
sudo apt install cmake ninja-build git git-lfs
```

#### Detialed building

```bash
git clone --recurse-submodules https://gitlab.freedesktop.org/monado/utilities/packaging/monado-data.git
cd monado-data
./make-package.sh
```

#### Installing

```bash
sudo dpkg -i build/*.deb
```
