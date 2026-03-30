#!/bin/bash
# Run from repo root
python3 -m grpc_tools.protoc -I./proto\
 --python_out=./py_generator \
 --pyi_out=./py_generator \
 --grpc_python_out=./py_generator \
 artificialdata.proto

python3 -m grpc_tools.protoc -I./proto \
  --python_out=./py_training \
  --pyi_out=./py_training \
  --grpc_python_out=./py_training \
  artificialdata_loader.proto
