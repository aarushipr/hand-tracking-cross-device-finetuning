https://stackoverflow.com/a/68964354

import pip
pip.main(['install', 'protobuf'])

import pip
pip.main(['install', 'grpcio'])


CXX=clang++-14 CC=clang-14 cmake .. -DCMAKE_CXX_FLAGS="-fuse-ld=mold" -DCMAKE_C_FLAGS="-fuse-ld=mold" -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo

CXX=clang++-13 CC=clang-13 cmake .. -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo


CXX=clang++-14 CC=clang-14 cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=1


cmake .. -DCMAKE_BUILD_TYPE=Debug -GNinja -DCMAKE_C_FLAGS="-g3 -O0 -fno-omit-frame-pointer" -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer"



XRT_FEATURE_TRACING=1