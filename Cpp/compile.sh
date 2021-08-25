#!/bin/bash
g++ mnist.cpp logger.cpp -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lnvinfer -L/usr/local/cuda/lib64 -lcudart -o Test