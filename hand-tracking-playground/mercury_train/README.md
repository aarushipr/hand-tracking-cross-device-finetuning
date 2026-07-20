# Training repository for Mercury hand tracking! (Experimental)

## Disclaimer

Mercury/Monado's hand tracking pipeline is stable and tested, and available within [Monado's codebase.](https://gitlab.freedesktop.org/monado/monado) If you just want to use our hand tracking with your VR headsets, that's where you'll want to go.

This repository is highly experimental, and hosts the *data annotation, artificial data generation, training, validation and evaluation code.* It's not used for anything in production. Use at your own risk.

## Building

For the C++ dataloader, run `setup.py install --user`. For anything else, it's a regular CMake build.

If you're missing dependencies for something, it's *generally easier to try disabling parts of the build.*
Depending on what you're doing, you may not need to build the whole thing, and you can get to what you're trying to do much faster.

## Running the data generator

Build the `data_generator` target and the `ad4_stereographic_projection` pybind extension (see Building, above), then run `./run_data_generator.sh`.
`data_generator.cpp`'s own defaults for the model manifest and output dir point at the original author's machine, so this script resolves them against the actual repo layout instead: `hand-tracking-playground/hand_scans/model_manifest.json` and a fresh `output_new/` dir, both siblings of `working-code/` at the thesis root.
Run it with no arguments first to see exactly what it resolved before anything starts; see the script's header comment for a low-cost smoke-test invocation, and for how to override any `GEN_*` variable.

**Heads up:** `GEN_SUPERROOT` gets wiped on every startup (the orchestrator clears it before listening), which is why the script defaults away from the existing `output/` dir.
Only point it there deliberately.

## Submodules

This repository uses Git submodules.
Many find these annoying.
The main secret for getting them to behave is *never to directly edit .gitmodules!*
Editing the file directly is tempting, but it desyncs the repository state with .gitmodules, which Git doesn't deal with very gracefully.
You can generally get what you need through `git mv`, `git rm`, or `git submodule add`.

## Style guide

For Markdown files, add a *single newline* between each sentence.
This makes documentation diffs much easier to read.

Ideally you'd be able to format the entire project using `meta/format-and-spellcheck.sh`, but not all of the formatters are hooked up yet, sorry.
C/C++ and CMake work at least.

We use clang-format for C/C++, cmake-format for CMake (todo: hint for how to install cmake-format, I think it's a Python package), markdownlint (todo: make a script for this, not just "install this VSCode extension") and pep8 (ditto) for Python.

## Contributions

Contributions are welcome!
Feel free to open merge requests on this repository, and I'll happily review/merge them!

Issues are tracked on this repository's issue page and in `doc/TODOS.md` depending on which is more convenient and what's actually being worked on.
