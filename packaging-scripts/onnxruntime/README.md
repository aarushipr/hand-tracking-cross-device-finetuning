# ONNX Runtime package maker

<!--
Copyright 2023, Collabora, Ltd.
SPDX-License-Identifier: MIT or BSL-1.0 or Apache-2.0
-->

A hack that lets you produce debian packages for ONNX Runtime from release tgz
files is CMake's CPack. Yes really CMake's CPack, it's silly and dumb but hey
it works and was fairly easy to do. Could probably extend to other distros, so
patches very much welcome. Run `./make-package.sh` to create the package.

#### Dependancies

Runtime dependancies
```bash
sudo apt install libstdc++6
```

"Compile" time dependancies
```bash
sudo apt install cmake ninja-build git git-lfs
```

#### Detailed instructions

```bash
git clone https://gitlab.freedesktop.org/monado/utilities/packaging/onnxruntime.git
cd onnxruntime
./make-package.sh
```

#### Installing

```bash
sudo dpkg -i build/*.deb
```

#### Proper packages

Here is where we list more proper packages for other distros:

* Arch [onnxruntime-bin](https://aur.archlinux.org/packages/onnxruntime-bin)
