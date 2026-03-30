# HMDHandRects annotator!

This is an internal tool that lets us semi-automatically annotate hand bounding boxes in EuRoC datasets collected from AR/VR headsets. 

# How to build
It's a regular CMake build - this is what I use:
```
mkdir build
cd build
cmake .. -GNinja \
    -DCMAKE_C_FLAGS="-g -march=native -O3 -fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS="-g -march=native -O3 -fno-omit-frame-pointer" \
    -DCMAKE_BUILD_TYPE="Debug" \
ninja
```

# How to use
* Collect a dataset using Monado's EuRoC recorder
* Run `build/src/machine_annotator --euroc_path=/path/to/euroc/dataset` to create initial NN annotations
* Run `build/src/human_annotator --euroc_path=/path/to/euroc/dataset` as-needed to correct/confirm those NN annotations
* Run `build/src/redactor --euroc_path=/path/to/euroc/dataset` as-needed to mark sensitive data for removal from your dataset
  * Then run `python3 py/apply_redactions.py --euroc_path=/path/to/euroc/dataset` to actually remove them

# Keyboard shortcuts for annotator
`p`: Confirm annotations in this frame; they'll be used during training

`o`: Un-confirm annotations in this frame; they will not be used during training

`s`: Swap handednesses in this image

`left click+drag`: Add a new bounding box to this image!

`right click`: Remove the bounding box your mouse is in 

`MMB/Scroll wheel`: Maps-style zoom/pan

`Left/Right arrow keys`: Scroll through frames
  * `Shift`: Skip to next boundary between confirmed/unconfirmed
  * `Ctrl`: Scroll by steps of five frames

`l`: start/end linking just one bounding box between frames. You'll see a `Linking!` modal when it's active.

  - `Esc`: Stops linking bounding boxes, takes you back to the default modality.

`i`: link all bounding boxes in this frame to the next confirmed frame (use with lots of care!)

`w`: Save the current annotations to `<euroc_path>/human_annotations_last.json`

# Many caveats
* Only works on 1280x800 images for now
* Only works on binocular/stereo datasets for now
* Keyboard shortcuts are hard-coded
* Haven't implemented file versioning yet. For now, manually git commit your updated annotations.json/human_annotated_last.json when you change it.

# Reference DataLoader
Is [here!]()

# If you need help, please contact me!
I'm very happy to help you get set up and understand the codebase. You can open an issue on this repo, find me on Monado's discord server (listed [here](https://monado.freedesktop.org/)), or [email me!](mailto:moses@collabora.com)


input:

i: link to next?

l: also links?

esc: stop linking?

p: confirm positions
ctrl+p: un-confirm positions

left/right arrow keys: switch frames

left click: draw box
right click: remove box

s: swap handednesses

w: save file

build:
```
cmake .. -GNinja -DBUILD_DOC=0 -DXRT_HAVE_BASALT_SLAM=0
```