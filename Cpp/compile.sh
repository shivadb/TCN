#!/bin/bash
g++ main.cpp logger.cpp -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lnvinfer -o Test